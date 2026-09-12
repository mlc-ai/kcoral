use std::time::Duration;

pub mod headers;
pub mod proto {
    tonic::include_proto!("kcoral.gateway.v1");
}
pub mod router;
pub mod supervisor;

pub const DEFAULT_MAX_REQUEST_BYTES: u64 = 256 * 1024 * 1024;

pub fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}

pub mod process;

pub fn validate_node_id(value: &str) -> anyhow::Result<()> {
    let valid = (1..=64).contains(&value.len())
        && value
            .bytes()
            .next()
            .is_some_and(|byte| byte.is_ascii_alphanumeric())
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_.-".contains(&byte));
    if !valid {
        anyhow::bail!("invalid node identifier {value:?}");
    }
    Ok(())
}

pub fn positive_duration(value: f64, name: &str) -> anyhow::Result<Duration> {
    if !value.is_finite() || value <= 0.0 {
        anyhow::bail!("{name} must be positive");
    }
    Ok(Duration::from_secs_f64(value))
}

pub fn nonnegative_duration(value: f64, name: &str) -> anyhow::Result<Duration> {
    if !value.is_finite() || value < 0.0 {
        anyhow::bail!("{name} must be non-negative");
    }
    Ok(Duration::from_secs_f64(value))
}

pub async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("install SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = terminate.recv() => {}
        }
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c()
        .await
        .expect("install Ctrl-C handler");
}
