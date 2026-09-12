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
