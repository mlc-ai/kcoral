use std::{ffi::OsString, str::FromStr, time::Duration};

use clap::Parser;
use kcoral::{
    init_tracing,
    supervisor::{run_server_lifecycle, RouterLinkConfig, ServerLifecycleConfig, SupervisorState},
};
use tokio::sync::watch;

#[derive(Debug, Parser)]
#[command(about = "Outbound control client and process supervisor for KCoral Server")]
struct Args {
    #[arg(long, env = "KCORAL_ROUTER_ENDPOINT")]
    router_endpoint: String,
    #[arg(long, env = "KCORAL_NODE_ID")]
    node_id: String,
    #[arg(long, env = "KCORAL_NODE_TOKEN", hide_env_values = true)]
    node_token: Option<String>,
    #[arg(
        long,
        default_value = "http://127.0.0.1:8000/",
        help = "local KCoral Server origin used only for health checks"
    )]
    server_url: String,
    #[arg(long, default_value_t = 2.0)]
    health_interval_seconds: f64,
    #[arg(long, default_value_t = 1.0)]
    health_timeout_seconds: f64,
    #[arg(long, default_value_t = 3)]
    failure_threshold: u32,
    #[arg(long, default_value_t = 30.0)]
    startup_grace_seconds: f64,
    #[arg(long, default_value_t = 60.0)]
    stable_reset_seconds: f64,
    #[arg(long, default_value_t = 5.0)]
    termination_grace_seconds: f64,
    #[arg(long, default_value_t = 1.0)]
    restart_min_delay_seconds: f64,
    #[arg(long, default_value_t = 30.0)]
    restart_max_delay_seconds: f64,
    #[arg(long, default_value_t = 0.2)]
    restart_jitter: f64,
    #[arg(long, default_value_t = 2.0)]
    control_connect_timeout_seconds: f64,
    #[arg(long, default_value_t = 1.0)]
    control_reconnect_min_delay_seconds: f64,
    #[arg(long, default_value_t = 30.0)]
    control_reconnect_max_delay_seconds: f64,
    #[arg(last = true, required = true)]
    command: Vec<OsString>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();
    let args = Args::parse();
    kcoral::process::adopt_server_descendants()?;
    let health_timeout = positive(args.health_timeout_seconds, "health timeout")?;
    let service = SupervisorState::new(
        args.node_id.clone(),
        reqwest::Url::from_str(&args.server_url)?,
        health_timeout,
    )?;
    let supervisor_config = ServerLifecycleConfig {
        health_interval: positive(args.health_interval_seconds, "health interval")?,
        failure_threshold: args.failure_threshold,
        startup_grace: nonnegative(args.startup_grace_seconds, "startup grace")?,
        stable_reset: nonnegative(args.stable_reset_seconds, "stable reset")?,
        termination_grace: nonnegative(args.termination_grace_seconds, "termination grace")?,
        restart_min_delay: positive(args.restart_min_delay_seconds, "minimum restart delay")?,
        restart_max_delay: positive(args.restart_max_delay_seconds, "maximum restart delay")?,
        restart_jitter: args.restart_jitter,
    };
    let control_config = RouterLinkConfig {
        router_endpoint: args.router_endpoint.clone(),
        node_token: args.node_token.clone(),
        heartbeat_interval: positive(args.health_interval_seconds, "control heartbeat")?,
        connect_timeout: positive(
            args.control_connect_timeout_seconds,
            "control connect timeout",
        )?,
        reconnect_min_delay: positive(
            args.control_reconnect_min_delay_seconds,
            "minimum control reconnect delay",
        )?,
        reconnect_max_delay: positive(
            args.control_reconnect_max_delay_seconds,
            "maximum control reconnect delay",
        )?,
    };
    let mut child_environment = vec![
        (
            OsString::from("KCORAL_ROUTER_ENDPOINT"),
            OsString::from(&args.router_endpoint),
        ),
        (
            OsString::from("KCORAL_NODE_ID"),
            OsString::from(&args.node_id),
        ),
    ];
    if let Some(token) = &args.node_token {
        child_environment.push((OsString::from("KCORAL_NODE_TOKEN"), OsString::from(token)));
    }

    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let signal_tx = shutdown_tx.clone();
    let signal_task = tokio::spawn(async move {
        shutdown_signal().await;
        let _ = signal_tx.send(true);
    });
    let mut lifecycle_task = tokio::spawn(run_server_lifecycle(
        supervisor_config,
        service.clone(),
        args.command,
        child_environment,
        shutdown_rx.clone(),
    ));
    let mut control_task = tokio::spawn(service.run_status_reporter(control_config, shutdown_rx));

    let result = tokio::select! {
        result = &mut lifecycle_task => {
            let _ = shutdown_tx.send(true);
            let control = control_task.await;
            result??;
            control??;
            Ok(())
        }
        result = &mut control_task => {
            let _ = shutdown_tx.send(true);
            lifecycle_task.await??;
            result??;
            Ok(())
        }
    };
    signal_task.abort();
    result
}

fn positive(value: f64, name: &str) -> anyhow::Result<Duration> {
    if !value.is_finite() || value <= 0.0 {
        anyhow::bail!("{name} must be positive");
    }
    Ok(Duration::from_secs_f64(value))
}

fn nonnegative(value: f64, name: &str) -> anyhow::Result<Duration> {
    if !value.is_finite() || value < 0.0 {
        anyhow::bail!("{name} must be non-negative");
    }
    Ok(Duration::from_secs_f64(value))
}

async fn shutdown_signal() {
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
