//! Durable jobs owned by the router, executed through ordinary node slots.
use super::{
    http::{forward_request, RequestTrace},
    job_store::{now, sync_directory, Admission, Job, JobConfig, Store},
    NodePool,
};
use axum::{
    body::Body,
    extract::{Path, State},
    http::{HeaderMap, HeaderValue, Request, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures_util::StreamExt;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, sync::Arc, time::Duration};
use tokio::{fs::File, io::AsyncWriteExt, sync::Semaphore};
use tokio_util::{io::ReaderStream, sync::CancellationToken, task::TaskTracker};
use tracing::error;
use uuid::Uuid;

#[derive(Clone)]
pub struct Jobs(Arc<Inner>);

struct Inner {
    store: Store,
    pool: NodePool,
    tasks: TaskTracker,
    shutdown: CancellationToken,
    uploads: Semaphore,
}

pub(super) struct Dispatch {
    pub jobs: Jobs,
    pub id: String,
}

impl Dispatch {
    pub async fn started(&self, node: &str, request_id: &str) -> anyhow::Result<()> {
        let mut job = self.jobs.0.store.get(self.id.clone()).await?.unwrap();
        job.state = "DISPATCHED".into();
        job.dispatched_at = Some(now());
        job.node_id = Some(node.into());
        job.request_id = Some(request_id.into());
        self.jobs.0.store.update(job).await
    }

    pub async fn cancelled(&self) {
        self.jobs.0.shutdown.cancelled().await;
    }
}

impl Jobs {
    pub async fn open(config: JobConfig, pool: NodePool) -> anyhow::Result<Self> {
        let store = tokio::task::spawn_blocking(|| Store::open(config)).await??;
        let jobs = Self(Arc::new(Inner {
            store,
            pool,
            tasks: TaskTracker::new(),
            shutdown: CancellationToken::new(),
            uploads: Semaphore::new(4),
        }));
        let records = jobs.0.store.all().await?;
        let ids: BTreeSet<_> = records.iter().map(|job| job.job_id.as_str()).collect();
        // A crash can leave an upload, unpublished output, or file whose DB commit failed.
        let mut files = tokio::fs::read_dir(&jobs.0.store.config.directory).await?;
        while let Some(file) = files.next_entry().await? {
            let name = file.file_name();
            let name = name.to_string_lossy();
            let orphan = name.starts_with(".upload-")
                || name.starts_with(".result-")
                || name
                    .strip_suffix(".request")
                    .or_else(|| name.strip_suffix(".response"))
                    .is_some_and(|id| !ids.contains(id));
            if orphan {
                tokio::fs::remove_file(file.path()).await?;
            }
        }
        let mut queued = Vec::new();
        for job in records {
            if job.state == "DISPATCHED" {
                jobs.fail(
                    job.job_id,
                    "execution_unknown",
                    "router restarted after dispatch; execution outcome is unknown",
                )
                .await?;
            } else if job.state == "QUEUED" {
                queued.push(job.job_id);
            }
        }
        jobs.cleanup().await?;
        for id in queued {
            jobs.spawn(id);
        }
        let cleaner = jobs.clone();
        jobs.0.tasks.spawn(async move {
            loop {
                tokio::select! {
                    _ = cleaner.0.shutdown.cancelled() => break,
                    _ = tokio::time::sleep(Duration::from_secs(1)) => {
                        if let Err(error) = cleaner.cleanup().await {
                            error!(%error, "job_cleanup_failed");
                        }
                    }
                }
            }
        });
        Ok(jobs)
    }

    pub fn routes(&self) -> Router {
        Router::new()
            .route("/jobs", post(submit))
            .route("/jobs/:id", get(status))
            .route("/jobs/:id/result", get(result))
            .with_state(self.clone())
    }

    pub fn begin_shutdown(&self) {
        self.0.shutdown.cancel();
        self.0.tasks.close();
    }

    pub async fn shutdown(&self) {
        self.begin_shutdown();
        self.0.tasks.wait().await;
    }

    fn spawn(&self, id: String) {
        let jobs = self.clone();
        self.0.tasks.spawn(async move {
            if let Err(error) = jobs.run(id.clone()).await {
                error!(job_id = id, %error, "job_execution_failed");
                if let Err(error) = jobs
                    .fail(
                        id.clone(),
                        "job_storage",
                        "could not save the job result; execution may have occurred",
                    )
                    .await
                {
                    error!(job_id = id, %error, "job_error_storage_failed");
                }
            }
        });
    }

    async fn cleanup(&self) -> anyhow::Result<()> {
        let cutoff = now().saturating_sub(self.0.store.config.retention.as_millis() as u64);
        for job in self.0.store.all().await? {
            if job.finished_at.is_some_and(|time| time <= cutoff) {
                self.0.store.expire(job).await?;
            }
        }
        Ok(())
    }

    async fn run(&self, id: String) -> anyhow::Result<()> {
        let job = self.0.store.get(id.clone()).await?.unwrap();
        let input = File::open(self.0.store.path(&id, "request")).await?;
        let length = input.metadata().await?.len();
        let mut request = Request::builder()
            .method("POST")
            .uri("/execute")
            .body(Body::from_stream(ReaderStream::new(input)))?;
        *request.headers_mut() = decode_headers(&job.request_headers)?;
        request
            .headers_mut()
            .insert("content-length", HeaderValue::from(length));
        let trace = RequestTrace::new();
        let dispatch = Dispatch {
            jobs: self.clone(),
            id: id.clone(),
        };
        let response = match forward_request(
            self.0.pool.clone(),
            request,
            trace.clone(),
            Some(dispatch),
        )
        .await
        {
            Ok(response) => response,
            Err(error) if error.is_shutdown() => return Ok(()),
            Err(error) => error.response(&trace),
        };
        let status = response.status();
        if status == StatusCode::BAD_GATEWAY {
            return self
                .fail(
                    id,
                    "execution_unknown",
                    "node connection failed; execution outcome is unknown",
                )
                .await;
        }
        let headers = response.headers().clone();
        let output = tempfile::Builder::new()
            .prefix(".result-")
            .tempfile_in(&self.0.store.config.directory)?;
        let saved = write_body(
            response.into_body(),
            &output,
            self.0.store.config.max_result_bytes,
        )
        .await;
        if let Err(error) = saved {
            let kind = if error.0 == StatusCode::PAYLOAD_TOO_LARGE {
                "response_too_large"
            } else {
                "execution_unknown"
            };
            return self.fail(id, kind, &error.2).await;
        }
        let metadata = read_result(output.path(), &headers).await;
        let metadata = match metadata {
            Ok(value)
                if matches!(
                    value["status"].as_str(),
                    Some("COMPLETED" | "FAILED" | "ERROR")
                ) =>
            {
                value
            }
            _ => {
                return self
                    .fail(
                        id,
                        "invalid_response",
                        "node returned an invalid job result",
                    )
                    .await
            }
        };
        let mut job = self.0.store.get(id.clone()).await?.unwrap();
        job.state = if status == StatusCode::OK {
            metadata["status"].as_str().unwrap().into()
        } else {
            "ERROR".into()
        };
        job.error = metadata.get("error").cloned();
        job.request_id = headers
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .map(String::from);
        job.response_status = status.as_u16();
        job.response_headers = encode_headers(&headers);
        self.finish(job, output).await
    }

    async fn fail(&self, id: String, kind: &str, message: &str) -> anyhow::Result<()> {
        let mut job = self.0.store.get(id).await?.unwrap();
        job.state = "ERROR".into();
        job.error = Some(json!({"kind": kind, "message": message}));
        job.response_status = match kind {
            "response_too_large" | "job_storage" => 500,
            _ => 502,
        };
        job.response_headers = vec![("content-type".into(), b"application/json".to_vec())];
        let body = serde_json::to_vec(
            &json!({"status": "ERROR", "error": job.error, "request_id": job.request_id}),
        )?;
        let output = tempfile::Builder::new()
            .prefix(".result-")
            .tempfile_in(&self.0.store.config.directory)?;
        write_body(Body::from(body), &output, u64::MAX)
            .await
            .map_err(|error| anyhow::anyhow!(error.2))?;
        self.finish(job, output).await
    }

    async fn finish(&self, mut job: Job, output: tempfile::NamedTempFile) -> anyhow::Result<()> {
        let path = self.0.store.path(&job.job_id, "response");
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            output.persist(&path)?;
            sync_directory(path.parent().unwrap())?;
            Ok(())
        })
        .await??;
        job.finished_at = Some(now());
        let input = self.0.store.path(&job.job_id, "request");
        self.0.store.update(job).await?;
        let _ = tokio::fs::remove_file(input).await;
        Ok(())
    }
}

struct ApiError(StatusCode, &'static str, String);

impl ApiError {
    fn invalid(message: impl ToString) -> Self {
        Self(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            message.to_string(),
        )
    }
    fn storage(error: impl std::fmt::Display) -> Self {
        error!(%error, "job_storage_failed");
        Self(
            StatusCode::INTERNAL_SERVER_ERROR,
            "job_storage",
            "job storage failed".into(),
        )
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut response = (
            self.0,
            Json(json!({"status": "ERROR", "error": {"kind": self.1, "message": self.2}})),
        )
            .into_response();
        if self.0 == StatusCode::SERVICE_UNAVAILABLE {
            response
                .headers_mut()
                .insert("retry-after", HeaderValue::from_static("1"));
        }
        response
    }
}

async fn submit(State(jobs): State<Jobs>, request: Request<Body>) -> Result<Response, ApiError> {
    if jobs.0.shutdown.is_cancelled() {
        return Err(ApiError(
            StatusCode::SERVICE_UNAVAILABLE,
            "router_busy",
            "router is shutting down".into(),
        ));
    }
    let _permit = jobs.0.uploads.try_acquire().map_err(|_| {
        ApiError(
            StatusCode::SERVICE_UNAVAILABLE,
            "router_busy",
            "too many uploads".into(),
        )
    })?;
    let (parts, body) = request.into_parts();
    let keys: Vec<_> = parts.headers.get_all("idempotency-key").iter().collect();
    let key = match keys.as_slice() {
        [] => None,
        [key] => Some(
            key.to_str()
                .ok()
                .filter(|key| {
                    !key.is_empty() && key.len() <= 128 && key.bytes().all(|b| b.is_ascii_graphic())
                })
                .ok_or_else(|| {
                    ApiError::invalid("Idempotency-Key must contain 1-128 visible ASCII characters")
                })?
                .to_string(),
        ),
        _ => return Err(ApiError::invalid("duplicate Idempotency-Key")),
    };
    let content_type = parts
        .headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| ApiError::invalid("missing Content-Type"))?;
    let boundary = multer::parse_boundary(content_type).map_err(ApiError::invalid)?;
    let input = tempfile::Builder::new()
        .prefix(".upload-")
        .tempfile_in(&jobs.0.store.config.directory)
        .map_err(ApiError::storage)?;
    write_body(body, &input, jobs.0.pool.max_request_bytes()).await?;
    let fingerprint = fingerprint(input.path(), boundary, &parts.headers).await?;
    let job = Job {
        job_id: Uuid::new_v4().to_string(),
        state: "QUEUED".into(),
        submitted_at: now(),
        dispatched_at: None,
        finished_at: None,
        node_id: None,
        request_id: None,
        error: None,
        request_headers: encode_headers(&crate::headers::forwarded(
            &parts.headers,
            &["idempotency-key", "content-length", "expect"],
        )),
        response_headers: vec![],
        response_status: 0,
    };
    // Admission and launching have one owner even if the submitting HTTP client disappears.
    let owner = jobs.clone();
    jobs.0
        .tasks
        .spawn(async move {
            let admission = owner
                .0
                .store
                .admit(job, key, fingerprint, input)
                .await
                .map_err(ApiError::storage)?;
            let job = match admission {
                Admission::Created(job) => {
                    owner.spawn(job.job_id.clone());
                    job
                }
                Admission::Existing(job) => job,
                Admission::Conflict => {
                    return Err(ApiError(
                        StatusCode::CONFLICT,
                        "idempotency_conflict",
                        "Idempotency-Key was used with different input".into(),
                    ))
                }
                Admission::Full => {
                    return Err(ApiError(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "router_busy",
                        "job store is full".into(),
                    ))
                }
            };
            let location = format!("/jobs/{}", job.job_id);
            Ok((StatusCode::ACCEPTED, [("location", location)], Json(job)).into_response())
        })
        .await
        .map_err(ApiError::storage)?
}

async fn lookup(jobs: &Jobs, id: String) -> Result<Job, ApiError> {
    let job = jobs.0.store.get(id).await.map_err(ApiError::storage)?;
    job.ok_or_else(|| {
        ApiError(
            StatusCode::NOT_FOUND,
            "job_not_found",
            "job is unknown or expired".into(),
        )
    })
}

async fn status(State(jobs): State<Jobs>, Path(id): Path<String>) -> Result<Json<Job>, ApiError> {
    Ok(Json(lookup(&jobs, id).await?))
}

async fn result(State(jobs): State<Jobs>, Path(id): Path<String>) -> Result<Response, ApiError> {
    let job = lookup(&jobs, id).await?;
    if job.pending() {
        return Ok((StatusCode::ACCEPTED, [("retry-after", "1")], Json(job)).into_response());
    }
    let file = File::open(jobs.0.store.path(&job.job_id, "response"))
        .await
        .map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                ApiError(
                    StatusCode::NOT_FOUND,
                    "job_not_found",
                    "job is unknown or expired".into(),
                )
            } else {
                ApiError::storage(error)
            }
        })?;
    let mut response = Response::new(Body::from_stream(ReaderStream::new(file)));
    *response.status_mut() =
        StatusCode::from_u16(job.response_status).map_err(ApiError::storage)?;
    *response.headers_mut() = decode_headers(&job.response_headers).map_err(ApiError::storage)?;
    Ok(response)
}

async fn write_body(
    body: Body,
    destination: &tempfile::NamedTempFile,
    limit: u64,
) -> Result<(), ApiError> {
    let mut file = File::from_std(destination.reopen().map_err(ApiError::storage)?);
    let mut stream = body.into_data_stream();
    let mut size = 0_u64;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(ApiError::invalid)?;
        size = size.saturating_add(chunk.len() as u64);
        if size > limit {
            return Err(ApiError(
                StatusCode::PAYLOAD_TOO_LARGE,
                "body_too_large",
                "body exceeds the configured size limit".into(),
            ));
        }
        file.write_all(&chunk).await.map_err(ApiError::storage)?;
    }
    file.sync_all().await.map_err(ApiError::storage)?;
    Ok(())
}

async fn fingerprint(
    path: &std::path::Path,
    boundary: String,
    headers: &HeaderMap,
) -> Result<String, ApiError> {
    let file = File::open(path).await.map_err(ApiError::storage)?;
    let mut multipart = multer::Multipart::new(ReaderStream::new(file), boundary);
    let mut program = None;
    let mut digest = Sha256::new();
    let mut blobs = BTreeSet::new();
    while let Some(mut field) = multipart.next_field().await.map_err(ApiError::invalid)? {
        let name = field.name().unwrap_or("").to_string();
        if name == "program" {
            if field.content_type().map(|value| value.as_ref()) != Some("application/json") {
                return Err(ApiError::invalid("program part must use application/json"));
            }
            if program.is_some() {
                return Err(ApiError::invalid("duplicate program part"));
            }
            let bytes = field.bytes().await.map_err(ApiError::invalid)?;
            // Hash original JSON bytes so distinct large integers or duplicate keys
            // cannot collapse into one idempotency fingerprint during JSON parsing.
            digest.update(&bytes);
            program = Some(serde_json::from_slice::<Value>(&bytes).map_err(ApiError::invalid)?);
        } else {
            if field.content_type().map(|value| value.as_ref()) != Some("application/octet-stream")
            {
                return Err(ApiError::invalid(
                    "blob parts must use application/octet-stream",
                ));
            }
            let hash = name
                .strip_prefix("blob:")
                .ok_or_else(|| ApiError::invalid("unexpected multipart part"))?;
            let mut digest = Sha256::new();
            while let Some(chunk) = field.chunk().await.map_err(ApiError::invalid)? {
                digest.update(chunk);
            }
            if format!("{:x}", digest.finalize()) != hash || !blobs.insert(hash.to_string()) {
                return Err(ApiError::invalid("invalid or duplicate blob hash"));
            }
        }
    }
    let program = program.ok_or_else(|| ApiError::invalid("missing program part"))?;
    let instructions = program["instructions"]
        .as_array()
        .filter(|v| !v.is_empty())
        .ok_or_else(|| ApiError::invalid("program needs a nonempty instructions array"))?;
    let mut referenced = BTreeSet::new();
    for instruction in instructions {
        if instruction["op"] == "upload" {
            if let Some(blob) = instruction.get("blob") {
                referenced.insert(
                    blob.as_str()
                        .ok_or_else(|| ApiError::invalid("invalid blob reference"))?
                        .to_string(),
                );
            }
        }
    }
    if referenced != blobs {
        return Err(ApiError::invalid(
            "submit must include exactly all referenced blobs",
        ));
    }
    digest.update(b"\0");
    for client in headers.get_all("x-kcoral-client") {
        digest.update(client.as_bytes());
        digest.update(b"\0");
    }
    Ok(format!("{:x}", digest.finalize()))
}

async fn read_result(path: &std::path::Path, headers: &HeaderMap) -> anyhow::Result<Value> {
    let content_type = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if content_type == "application/json" {
        return Ok(serde_json::from_slice(&tokio::fs::read(path).await?)?);
    }
    let boundary = multer::parse_boundary(content_type)?;
    let mut multipart =
        multer::Multipart::new(ReaderStream::new(File::open(path).await?), boundary);
    while let Some(field) = multipart.next_field().await? {
        if field.name() == Some("result") {
            return Ok(serde_json::from_slice(&field.bytes().await?)?);
        }
    }
    anyhow::bail!("missing result part")
}

fn encode_headers(headers: &HeaderMap) -> Vec<(String, Vec<u8>)> {
    headers
        .iter()
        .map(|(name, value)| (name.to_string(), value.as_bytes().to_vec()))
        .collect()
}

fn decode_headers(headers: &[(String, Vec<u8>)]) -> anyhow::Result<HeaderMap> {
    let mut result = HeaderMap::new();
    for (name, value) in headers {
        result.append(
            name.parse::<http::HeaderName>()?,
            HeaderValue::from_bytes(value)?,
        );
    }
    Ok(result)
}
