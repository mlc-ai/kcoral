use anyhow::Context;
use fs2::FileExt;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Debug)]
pub struct JobConfig {
    pub directory: PathBuf,
    pub retention: Duration,
    pub max_jobs: usize,
    pub max_result_bytes: u64,
}

#[derive(Clone, Serialize, Deserialize)]
pub(super) struct Job {
    pub job_id: String,
    pub state: String,
    pub submitted_at: u64,
    pub dispatched_at: Option<u64>,
    pub finished_at: Option<u64>,
    pub node_id: Option<String>,
    pub request_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<serde_json::Value>,
    #[serde(skip)]
    pub request_headers: Vec<(String, Vec<u8>)>,
    #[serde(skip)]
    pub response_headers: Vec<(String, Vec<u8>)>,
    #[serde(skip)]
    pub response_status: u16,
}

impl Job {
    pub fn pending(&self) -> bool {
        matches!(self.state.as_str(), "QUEUED" | "DISPATCHED")
    }
}

pub(super) struct Store {
    pub config: JobConfig,
    connection: Arc<Mutex<Connection>>,
    // Held for the lifetime of all requests and background jobs using this store.
    _lock: File,
}

pub(super) enum Admission {
    Created(Job),
    Existing(Job),
    Conflict,
    Full,
}

pub(super) fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

impl Store {
    pub fn open(config: JobConfig) -> anyhow::Result<Self> {
        anyhow::ensure!(config.max_jobs > 0, "max jobs must be positive");
        anyhow::ensure!(
            config.retention.as_millis() > 0 && config.retention.as_millis() <= u64::MAX as u128,
            "job retention must be between one millisecond and u64::MAX milliseconds"
        );
        anyhow::ensure!(
            config.max_result_bytes > 0,
            "max job result bytes must be positive"
        );
        std::fs::create_dir_all(&config.directory)?;
        let lock = File::options()
            .create(true)
            .truncate(false)
            .write(true)
            .open(config.directory.join("lock"))?;
        lock.try_lock_exclusive()
            .context("job directory is already in use")?;
        let connection = Connection::open(config.directory.join("jobs.sqlite3"))?;
        connection.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=FULL;
             CREATE TABLE IF NOT EXISTS jobs (
                 id TEXT PRIMARY KEY, key TEXT UNIQUE, fingerprint TEXT NOT NULL,
                 record TEXT NOT NULL, request_headers TEXT NOT NULL,
                 response_headers TEXT NOT NULL, response_status INTEGER NOT NULL
             );",
        )?;
        Ok(Self {
            config,
            connection: Arc::new(Mutex::new(connection)),
            _lock: lock,
        })
    }

    async fn access<T: Send + 'static>(
        &self,
        operation: impl FnOnce(&mut Connection) -> anyhow::Result<T> + Send + 'static,
    ) -> anyhow::Result<T> {
        let connection = self.connection.clone();
        tokio::task::spawn_blocking(move || operation(&mut connection.lock().unwrap())).await?
    }

    pub fn path(&self, id: &str, extension: &str) -> PathBuf {
        self.config.directory.join(format!("{id}.{extension}"))
    }

    pub async fn get(&self, id: String) -> anyhow::Result<Option<Job>> {
        self.access(move |db| {
            let mut statement = db.prepare("SELECT * FROM jobs WHERE id = ?1")?;
            Ok(statement.query_row([id], read_job).optional()?)
        })
        .await
    }

    pub async fn all(&self) -> anyhow::Result<Vec<Job>> {
        self.access(|db| {
            let mut statement = db.prepare("SELECT * FROM jobs ORDER BY rowid")?;
            let jobs = statement
                .query_map([], read_job)?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(jobs)
        })
        .await
    }

    pub async fn admit(
        &self,
        job: Job,
        key: Option<String>,
        fingerprint: String,
        input: tempfile::NamedTempFile,
    ) -> anyhow::Result<Admission> {
        let max_jobs = self.config.max_jobs;
        let path = self.path(&job.job_id, "request");
        self.access(move |db| {
            let transaction = db.transaction()?;
            if let Some(ref key) = key {
                let existing = transaction
                    .query_row("SELECT * FROM jobs WHERE key = ?1", [key], |row| {
                        Ok((read_job(row)?, row.get::<_, String>(2)?))
                    })
                    .optional()?;
                if let Some((existing, digest)) = existing {
                    return Ok(if digest == fingerprint {
                        Admission::Existing(existing)
                    } else {
                        Admission::Conflict
                    });
                }
            }
            let count: usize =
                transaction.query_row("SELECT count(*) FROM jobs", [], |row| row.get(0))?;
            if count >= max_jobs {
                return Ok(Admission::Full);
            }
            input.persist(&path)?;
            sync_directory(path.parent().unwrap())?;
            let saved = transaction
                .execute(
                    "INSERT INTO jobs VALUES (?1, ?2, ?3, ?4, ?5, '[]', 0)",
                    params![
                        job.job_id,
                        key,
                        fingerprint,
                        serde_json::to_string(&job)?,
                        serde_json::to_string(&job.request_headers)?
                    ],
                )
                .and_then(|_| transaction.commit());
            if let Err(error) = saved {
                let _ = std::fs::remove_file(path);
                return Err(error.into());
            }
            Ok(Admission::Created(job))
        })
        .await
    }

    pub async fn update(&self, job: Job) -> anyhow::Result<()> {
        self.access(move |db| {
            db.execute(
                "UPDATE jobs SET record = ?2, response_headers = ?3, response_status = ?4 WHERE id = ?1",
                params![job.job_id, serde_json::to_string(&job)?,
                        serde_json::to_string(&job.response_headers)?, job.response_status],
            )?;
            Ok(())
        }).await
    }

    pub async fn expire(&self, job: Job) -> anyhow::Result<()> {
        let input = self.path(&job.job_id, "request");
        let output = self.path(&job.job_id, "response");
        self.access(move |db| {
            // Serialize deletion with lookup/admission. Open downloads retain their file handle.
            for path in [input, output] {
                match std::fs::remove_file(path) {
                    Ok(()) => (),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
                    Err(error) => return Err(error.into()),
                }
            }
            db.execute("DELETE FROM jobs WHERE id = ?1", [job.job_id])?;
            Ok(())
        })
        .await
    }
}

fn read_job(row: &rusqlite::Row<'_>) -> rusqlite::Result<Job> {
    fn decode<T: serde::de::DeserializeOwned>(
        row: &rusqlite::Row<'_>,
        index: usize,
    ) -> rusqlite::Result<T> {
        let text: String = row.get(index)?;
        serde_json::from_str(&text).map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(
                index,
                rusqlite::types::Type::Text,
                Box::new(error),
            )
        })
    }
    let mut job: Job = decode(row, 3)?;
    job.request_headers = decode(row, 4)?;
    job.response_headers = decode(row, 5)?;
    job.response_status = row.get(6)?;
    Ok(job)
}

pub(super) fn sync_directory(path: &Path) -> std::io::Result<()> {
    File::open(path)?.sync_all()
}
