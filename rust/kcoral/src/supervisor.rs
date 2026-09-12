use std::{
    collections::HashMap,
    ffi::OsString,
    process::Stdio,
    sync::Arc,
    time::{Duration, Instant},
};

use async_stream::stream;
use nix::{
    sys::signal::{kill, Signal},
    unistd::Pid,
};
use rand::Rng;
use reqwest::Url;
use serde::Deserialize;
use serde_json::Value;
use tokio::{
    process::Child,
    sync::{watch, RwLock},
};
use tonic::{metadata::MetadataValue, transport::Endpoint, Request};
use tracing::{info, warn};
use uuid::Uuid;

use crate::proto::{router_gateway_client::RouterGatewayClient, SupervisorStatus};
use crate::validate_node_id;

#[derive(Clone, Debug)]
pub struct ServerLifecycleConfig {
    pub health_interval: Duration,
    pub failure_threshold: u32,
    pub startup_grace: Duration,
    pub stable_reset: Duration,
    pub termination_grace: Duration,
    pub restart_min_delay: Duration,
    pub restart_max_delay: Duration,
    pub restart_jitter: f64,
}

impl ServerLifecycleConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.health_interval.is_zero()
            || self.failure_threshold == 0
            || self.restart_min_delay.is_zero()
            || self.restart_max_delay.is_zero()
        {
            anyhow::bail!("health intervals, thresholds, and restart delays must be positive");
        }
        if self.restart_min_delay > self.restart_max_delay {
            anyhow::bail!("minimum restart delay cannot exceed maximum restart delay");
        }
        if !self.restart_jitter.is_finite() || !(0.0..=0.5).contains(&self.restart_jitter) {
            anyhow::bail!("restart jitter must be between 0 and 0.5");
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct RouterLinkConfig {
    pub router_endpoint: String,
    pub node_token: Option<String>,
    pub heartbeat_interval: Duration,
    pub connect_timeout: Duration,
    pub reconnect_min_delay: Duration,
    pub reconnect_max_delay: Duration,
}

impl RouterLinkConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        let endpoint = http::Uri::try_from(self.router_endpoint.as_str())?;
        if !matches!(endpoint.scheme_str(), Some("http" | "https"))
            || endpoint.authority().is_none()
            || endpoint
                .path_and_query()
                .is_some_and(|path| path.as_str() != "/")
        {
            anyhow::bail!("router endpoint must be an HTTP or HTTPS origin");
        }
        if self.heartbeat_interval.is_zero()
            || self.connect_timeout.is_zero()
            || self.reconnect_min_delay.is_zero()
            || self.reconnect_max_delay.is_zero()
        {
            anyhow::bail!("control connection intervals must be positive");
        }
        if self.reconnect_min_delay > self.reconnect_max_delay {
            anyhow::bail!("minimum reconnect delay cannot exceed maximum reconnect delay");
        }
        if self.node_token.as_deref() == Some("") {
            anyhow::bail!("node token cannot be empty");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
struct HealthSnapshot {
    status: SupervisorStatus,
    observed_at: Option<Instant>,
}

#[derive(Clone)]
pub struct SupervisorState {
    node_id: Arc<str>,
    server_url: Url,
    health_timeout: Duration,
    client: reqwest::Client,
    supervisor_instance_id: Arc<str>,
    health: Arc<RwLock<HealthSnapshot>>,
    updates: watch::Sender<u64>,
}

impl SupervisorState {
    pub fn new(node_id: String, server_url: Url, health_timeout: Duration) -> anyhow::Result<Self> {
        validate_node_id(&node_id)?;
        if !matches!(server_url.scheme(), "http" | "https") {
            anyhow::bail!("server URL must use http or https");
        }
        if server_url.cannot_be_a_base()
            || server_url.path() != "/"
            || server_url.query().is_some()
            || server_url.fragment().is_some()
        {
            anyhow::bail!("server URL must be an HTTP origin");
        }
        if health_timeout.is_zero() {
            anyhow::bail!("health timeout must be positive");
        }
        let client = reqwest::Client::builder()
            .connect_timeout(health_timeout)
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .build()?;
        let (updates, _) = watch::channel(0);
        Ok(Self {
            node_id: Arc::from(node_id),
            server_url,
            health_timeout,
            client,
            supervisor_instance_id: Arc::from(Uuid::new_v4().to_string()),
            health: Arc::new(RwLock::new(HealthSnapshot::default())),
            updates,
        })
    }

    pub async fn refresh_health(&self) -> anyhow::Result<()> {
        let health_url = self.server_url.join("health")?;
        let result = tokio::time::timeout(self.health_timeout, async {
            let response = self
                .client
                .get(health_url)
                .send()
                .await?
                .error_for_status()?;
            parse_health(response.json::<Value>().await?)
        })
        .await;

        let parsed = match result {
            Ok(Ok(parsed)) => parsed,
            Ok(Err(error)) => return self.record_health_error(error.to_string()).await,
            Err(_) => {
                return self
                    .record_health_error("local server health check timed out".to_string())
                    .await
            }
        };
        let observed_at = Some(Instant::now());
        let mut health = self.health.write().await;
        let old_instance = &health.status.server_instance_id;
        if !old_instance.is_empty() && *old_instance != parsed.server_instance_id {
            info!(
                old_instance,
                new_instance = parsed.server_instance_id,
                "local KCoral Server instance changed"
            );
        }
        *health = HealthSnapshot {
            status: parsed,
            observed_at,
        };
        drop(health);
        self.notify_status();
        Ok(())
    }

    async fn record_health_error(&self, error: String) -> anyhow::Result<()> {
        self.mark_server_unavailable(&error).await;
        anyhow::bail!(error)
    }

    async fn mark_server_unavailable(&self, error: &str) {
        let mut health = self.health.write().await;
        health.status.server_healthy = false;
        health.status.last_error = error.to_string();
        drop(health);
        self.notify_status();
    }

    fn notify_status(&self) {
        self.updates.send_modify(|generation| {
            *generation = generation.wrapping_add(1);
        });
    }

    pub async fn current_status(&self) -> SupervisorStatus {
        let health = self.health.read().await;
        SupervisorStatus {
            node_id: self.node_id.to_string(),
            supervisor_instance_id: self.supervisor_instance_id.to_string(),
            health_age_millis: health
                .observed_at
                .map(|observed| {
                    observed
                        .elapsed()
                        .as_millis()
                        .try_into()
                        .unwrap_or(u64::MAX)
                })
                .unwrap_or(u64::MAX),
            ..health.status.clone()
        }
    }

    pub async fn run_status_reporter(
        self,
        config: RouterLinkConfig,
        mut shutdown: watch::Receiver<bool>,
    ) -> anyhow::Result<()> {
        config.validate()?;
        let endpoint = Endpoint::from_shared(config.router_endpoint.clone())?
            .connect_timeout(config.connect_timeout)
            .http2_keep_alive_interval(Duration::from_secs(15))
            .keep_alive_timeout(Duration::from_secs(5))
            .keep_alive_while_idle(true);
        let mut backoff = Backoff::new(config.reconnect_min_delay, config.reconnect_max_delay);

        loop {
            if *shutdown.borrow() {
                return Ok(());
            }
            let result = async {
                let channel = endpoint.clone().connect().await?;
                let mut client = RouterGatewayClient::new(channel);
                let service = self.clone();
                let mut updates = self.updates.subscribe();
                let heartbeat_interval = config.heartbeat_interval;
                let outbound = stream! {
                    yield service.current_status().await;
                    let mut heartbeat = tokio::time::interval(heartbeat_interval);
                    heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                    // Consume the immediate first tick; the initial status above already covers it.
                    heartbeat.tick().await;
                    loop {
                        tokio::select! {
                            _ = heartbeat.tick() => yield service.current_status().await,
                            changed = updates.changed() => {
                                if changed.is_err() {
                                    break;
                                }
                                yield service.current_status().await;
                            }
                        }
                    }
                };
                let mut request = Request::new(outbound);
                if let Some(token) = &config.node_token {
                    let value = MetadataValue::try_from(format!("Bearer {token}"))?;
                    request.metadata_mut().insert("authorization", value);
                }
                let mut commands = client.connect_supervisor(request).await?.into_inner();
                let first = commands
                    .message()
                    .await?
                    .ok_or_else(|| anyhow::anyhow!("router closed the control stream"))?;
                let _ = first;
                backoff.reset();
                info!(
                    node = self.node_id.as_ref(),
                    router = config.router_endpoint,
                    "connected outbound supervisor control stream"
                );
                loop {
                    commands
                        .message()
                        .await?
                        .ok_or_else(|| anyhow::anyhow!("router closed the supervisor stream"))?;
                }
                #[allow(unreachable_code)]
                Ok::<(), anyhow::Error>(())
            };
            let result = tokio::select! {
                result = result => result,
                _ = shutdown.changed() => return Ok(()),
            };

            if *shutdown.borrow() {
                return Ok(());
            }
            if let Err(error) = result {
                let delay = backoff.next_delay();
                warn!(
                    node = self.node_id.as_ref(),
                    %error,
                    delay_seconds = delay.as_secs_f64(),
                    "outbound supervisor control stream disconnected"
                );
                tokio::select! {
                    _ = tokio::time::sleep(delay) => {}
                    changed = shutdown.changed() => {
                        if changed.is_err() || *shutdown.borrow() {
                            return Ok(());
                        }
                    }
                }
            }
        }
    }
}

pub async fn run_server_lifecycle(
    config: ServerLifecycleConfig,
    service: SupervisorState,
    command: Vec<OsString>,
    child_environment: Vec<(OsString, OsString)>,
    mut shutdown: watch::Receiver<bool>,
) -> anyhow::Result<()> {
    config.validate()?;
    if command.is_empty() {
        anyhow::bail!("a child command is required after --");
    }
    let mut backoff = Backoff::new(config.restart_min_delay, config.restart_max_delay);

    loop {
        if *shutdown.borrow() {
            return Ok(());
        }
        service
            .mark_server_unavailable("KCoral Server is starting")
            .await;
        let mut child = spawn_server(&command, &child_environment)?;
        let pid = child.id().unwrap_or_default();
        let tree = crate::process::ProcessTree::new(pid)?;
        info!(pid, "started KCoral Server child");
        let started = Instant::now();
        let mut healthy_since = None;
        let mut failures = 0_u32;
        let mut interval = tokio::time::interval(config.health_interval);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let restart_reason = loop {
            tokio::select! {
                status = child.wait() => {
                    let reason = format!("child exited with {}", status?);
                    tree.finish().await?;
                    service.mark_server_unavailable(&reason).await;
                    warn!(pid, %reason, "KCoral Server child stopped");
                    break reason;
                }
                _ = interval.tick() => {
                    match service.refresh_health().await {
                        Ok(()) => {
                            failures = 0;
                            let healthy_since = healthy_since.get_or_insert_with(Instant::now);
                            if healthy_since.elapsed() >= config.stable_reset {
                                backoff.reset();
                            }
                        }
                        Err(error) => {
                            healthy_since = None;
                            if started.elapsed() < config.startup_grace {
                                warn!(pid, %error, "health check failed during startup grace period");
                                continue;
                            }
                            failures = failures.saturating_add(1);
                            warn!(pid, failures, threshold = config.failure_threshold, %error, "KCoral Server health check failed");
                            if failures >= config.failure_threshold {
                                let reason = format!("health failed {failures} consecutive times: {error}");
                                terminate_server(&mut child, &tree, config.termination_grace).await?;
                                break reason;
                            }
                        }
                    }
                }
                changed = shutdown.changed() => {
                    if changed.is_err() || *shutdown.borrow() {
                        info!(pid, "stopping KCoral Server child");
                        service.mark_server_unavailable("Supervisor is shutting down").await;
                        terminate_server(&mut child, &tree, config.termination_grace).await?;
                        return Ok(());
                    }
                }
            }
        };
        tree.finish().await?;

        service.mark_server_unavailable(&restart_reason).await;
        let base_delay = backoff.next_delay();
        let delay = jittered(base_delay, config.restart_jitter);
        warn!(
            reason = restart_reason,
            delay_seconds = delay.as_secs_f64(),
            "restarting KCoral Server after backoff"
        );
        tokio::select! {
            _ = tokio::time::sleep(delay) => {}
            changed = shutdown.changed() => {
                if changed.is_err() || *shutdown.borrow() {
                    return Ok(());
                }
            }
        }
    }
}

fn parse_health(body: Value) -> anyhow::Result<SupervisorStatus> {
    let object = body
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("health response is not an object"))?;
    if object.get("status").and_then(Value::as_str) != Some("ok") {
        anyhow::bail!("health response status is not ok");
    }
    let server_instance_id = required_string(object.get("instance_id"), "instance_id")?;
    let workers = object
        .get("workers")
        .and_then(Value::as_array)
        .filter(|workers| !workers.is_empty())
        .ok_or_else(|| anyhow::anyhow!("health response has no workers"))?;
    let busy_workers = workers
        .iter()
        .filter(|worker| worker.get("status").and_then(Value::as_str) == Some("busy"))
        .count() as u64;
    if workers.iter().any(|worker| {
        !matches!(
            worker.get("status").and_then(Value::as_str),
            Some("idle" | "busy")
        )
    }) {
        anyhow::bail!("health response has invalid worker status");
    }
    Ok(SupervisorStatus {
        server_healthy: true,
        server_instance_id,
        worker_count: workers.len() as u64,
        busy_workers,
        target: required_string_map(object.get("target"), "target")?,
        versions: required_string_map(object.get("versions"), "versions")?,
        ..SupervisorStatus::default()
    })
}

fn required_string(value: Option<&Value>, name: &str) -> anyhow::Result<String> {
    value
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| anyhow::anyhow!("health response has invalid {name}"))
}

fn required_string_map(
    value: Option<&Value>,
    name: &str,
) -> anyhow::Result<HashMap<String, String>> {
    HashMap::deserialize(value.unwrap_or(&Value::Null))
        .map_err(|_| anyhow::anyhow!("health response has invalid {name}"))
}

fn spawn_server(
    command: &[OsString],
    environment: &[(OsString, OsString)],
) -> anyhow::Result<Child> {
    let mut process = tokio::process::Command::new(&command[0]);
    process
        .args(&command[1..])
        .envs(environment.iter().cloned())
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .process_group(0);
    Ok(process.spawn()?)
}

async fn terminate_server(
    child: &mut Child,
    tree: &crate::process::ProcessTree,
    grace: Duration,
) -> anyhow::Result<()> {
    let Some(pid) = child.id() else {
        return Ok(());
    };
    match kill(Pid::from_raw(pid as i32), Signal::SIGTERM) {
        Ok(()) | Err(nix::errno::Errno::ESRCH) => {}
        Err(error) => return Err(error.into()),
    }
    if let Ok(status) = tokio::time::timeout(grace, child.wait()).await {
        status?;
    } else {
        warn!(pid, "child did not stop after SIGTERM; sending SIGKILL");
        tree.signal(nix::libc::SIGKILL)?;
        child.start_kill()?;
        child.wait().await?;
    }
    tree.finish().await?;
    Ok(())
}

#[derive(Clone, Debug)]
struct Backoff {
    initial: Duration,
    maximum: Duration,
    next: Duration,
}

impl Backoff {
    fn new(initial: Duration, maximum: Duration) -> Self {
        Self {
            initial,
            maximum,
            next: initial,
        }
    }

    fn next_delay(&mut self) -> Duration {
        let current = self.next;
        self.next = self.next.saturating_mul(2).min(self.maximum);
        current
    }

    fn reset(&mut self) {
        self.next = self.initial;
    }
}

fn jittered(base: Duration, fraction: f64) -> Duration {
    if fraction == 0.0 {
        return base;
    }
    let factor = rand::rng().random_range((1.0 - fraction)..=(1.0 + fraction));
    Duration::from_secs_f64(base.as_secs_f64() * factor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_is_bounded_and_resets() {
        let mut backoff = Backoff::new(Duration::from_secs(1), Duration::from_secs(5));
        assert_eq!(backoff.next_delay(), Duration::from_secs(1));
        assert_eq!(backoff.next_delay(), Duration::from_secs(2));
        assert_eq!(backoff.next_delay(), Duration::from_secs(4));
        assert_eq!(backoff.next_delay(), Duration::from_secs(5));
        assert_eq!(backoff.next_delay(), Duration::from_secs(5));
        backoff.reset();
        assert_eq!(backoff.next_delay(), Duration::from_secs(1));
    }

    #[test]
    fn zero_jitter_is_deterministic() {
        assert_eq!(
            jittered(Duration::from_millis(250), 0.0),
            Duration::from_millis(250)
        );
    }

    #[test]
    fn parses_server_health() {
        let health = parse_health(serde_json::json!({
            "status": "ok",
            "instance_id": "server-1",
            "active_requests": 2,
            "queue_length": 1,
            "gpu_count": 1,
            "target": {"arch": "sm_100a"},
            "versions": {"cuda": "13.0"},
            "workers": [
                {"status": "busy"},
                {"status": "idle"}
            ]
        }))
        .unwrap();
        assert!(health.server_healthy);
        assert_eq!(health.worker_count, 2);
        assert_eq!(health.busy_workers, 1);
    }

    #[test]
    fn validates_outbound_control_configuration() {
        let config = RouterLinkConfig {
            router_endpoint: "https://router.example.com/".to_string(),
            node_token: Some("secret".to_string()),
            heartbeat_interval: Duration::from_secs(2),
            connect_timeout: Duration::from_secs(2),
            reconnect_min_delay: Duration::from_secs(1),
            reconnect_max_delay: Duration::from_secs(30),
        };
        assert!(config.validate().is_ok());
    }
}
