//! HTTP dispatch, outbound gRPC registration, and node scheduling.
mod gateway;
mod http;
mod pool;

pub use http::{app, serve};
use pool::{AcquireError, SlotGuard};
pub use pool::{NodePool, RouterConfig};
use std::time::Duration;

const NODE_HEADER: &str = "x-kcoral-node";
const PROTOCOL_VERSION: u32 = 1;
pub const DATA_CHUNK_BYTES: usize = 256 * 1024;
const GRPC_MESSAGE_BYTES: usize = DATA_CHUNK_BYTES + 64 * 1024;
const FIRST_FRAME_TIMEOUT: Duration = Duration::from_secs(10);
