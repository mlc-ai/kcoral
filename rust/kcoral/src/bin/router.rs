use std::net::SocketAddr;

use clap::Parser;
use kcoral::{
    init_tracing, nonnegative_duration, positive_duration,
    router::{serve, NodePool, RouterConfig},
    shutdown_signal, DEFAULT_MAX_REQUEST_BYTES,
};
use tokio::sync::watch;

#[derive(Debug, Parser)]
#[command(about = "HTTP and outbound-node gRPC router for a KCoral cohort")]
struct Args {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 9000)]
    port: u16,
    #[arg(long, default_value_t = 1.0)]
    status_check_interval_seconds: f64,
    #[arg(long, default_value_t = 5.0)]
    max_status_age_seconds: f64,
    #[arg(long, default_value_t = 600.0)]
    node_retention_seconds: f64,
    #[arg(long, default_value_t = 3)]
    unhealthy_threshold: u32,
    #[arg(long, default_value_t = 2)]
    recovery_threshold: u32,
    #[arg(long, default_value_t = 1800.0)]
    queue_wait_timeout_seconds: f64,
    #[arg(long, default_value_t = 1024)]
    max_queued_requests: usize,
    #[arg(long, default_value_t = DEFAULT_MAX_REQUEST_BYTES)]
    max_request_bytes: u64,
    #[arg(long, env = "KCORAL_NODE_TOKEN", hide_env_values = true)]
    node_token: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();
    let args = Args::parse();
    let listen: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    let config = RouterConfig {
        status_check_interval: positive_duration(
            args.status_check_interval_seconds,
            "status check interval",
        )?,
        max_status_age: positive_duration(args.max_status_age_seconds, "maximum status age")?,
        node_retention: positive_duration(args.node_retention_seconds, "node retention")?,
        unhealthy_threshold: args.unhealthy_threshold,
        recovery_threshold: args.recovery_threshold,
        queue_wait_timeout: nonnegative_duration(
            args.queue_wait_timeout_seconds,
            "queue wait timeout",
        )?,
        max_queued_requests: args.max_queued_requests,
        max_request_bytes: args.max_request_bytes,
        node_token: args.node_token,
    };
    let pool = NodePool::new(config)?;
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let health_task = tokio::spawn(pool.clone().run_status_loop(shutdown_rx));
    serve(listen, pool, shutdown_signal()).await?;
    let _ = shutdown_tx.send(true);
    health_task.await?;
    Ok(())
}
