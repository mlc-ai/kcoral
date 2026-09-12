use super::{
    pool::{validate_node_id, Slot},
    AcquireError, NodePool, SlotGuard, DATA_CHUNK_BYTES, NODE_HEADER,
};
use crate::{
    headers,
    proto::{slot_frame::Payload, EndOfBody, HttpHeader, RequestHead, SlotFrame},
};
use async_stream::try_stream;
use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, HeaderName, HeaderValue, Request as HttpRequest, Response as HttpResponse},
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use http::StatusCode;
use serde_json::{json, Value};
use std::{
    io,
    net::SocketAddr,
    sync::{atomic::Ordering as AtomicOrdering, Arc},
    time::Instant,
};
use tonic::Status;
use tracing::info;
use uuid::Uuid;
pub fn app(pool: NodePool) -> Router {
    let http = Router::new()
        .route("/health", get(health_handler))
        .route("/execute", post(execute_handler))
        .with_state(pool.clone());
    http.merge(super::gateway::routes(pool))
}

async fn health_handler(State(pool): State<NodePool>) -> Response {
    let snapshot = pool.snapshot().await;
    let status = if snapshot.get("status").and_then(Value::as_str) == Some("ok") {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(snapshot)).into_response()
}

// Request accounting follows the response body, including cancellation.
struct RequestTrace {
    id: String,
    started: Instant,
    state: std::sync::Mutex<TraceState>,
    sent: std::sync::atomic::AtomicU64,
    received: std::sync::atomic::AtomicU64,
}

struct TraceState {
    node: Option<String>,
    status: u16,
    queue_ms: u128,
    queued_at: Option<Instant>,
    reason: &'static str,
}

impl RequestTrace {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            id: Uuid::new_v4().to_string(),
            started: Instant::now(),
            state: std::sync::Mutex::new(TraceState {
                node: None,
                status: 0,
                queue_ms: 0,
                queued_at: None,
                reason: "client_disconnected",
            }),
            sent: std::sync::atomic::AtomicU64::new(0),
            received: std::sync::atomic::AtomicU64::new(0),
        })
    }
    fn finish(&self, status: u16, reason: &'static str) {
        let mut state = self.state.lock().unwrap();
        state.status = status;
        state.reason = reason;
    }
}

impl Drop for RequestTrace {
    fn drop(&mut self) {
        let state = self.state.lock().unwrap();
        info!(
            request_id = self.id,
            node = state.node.as_deref().unwrap_or(""),
            http_status = state.status,
            queue_wait_ms = state
                .queued_at
                .map_or(state.queue_ms, |t| t.elapsed().as_millis())
                as u64,
            duration_ms = self.started.elapsed().as_millis() as u64,
            request_bytes = self.sent.load(AtomicOrdering::Relaxed),
            response_bytes = self.received.load(AtomicOrdering::Relaxed),
            finish_reason = state.reason,
            "request_finished"
        );
    }
}

#[derive(Debug)]
struct UploadError {
    reason: &'static str,
    status: StatusCode,
    kind: &'static str,
    message: &'static str,
}

struct UploadTask(tokio::task::JoinHandle<Result<(), UploadError>>);
impl Drop for UploadTask {
    fn drop(&mut self) {
        self.0.abort();
    }
}

async fn execute_handler(State(pool): State<NodePool>, request: HttpRequest<Body>) -> Response {
    let trace = RequestTrace::new();
    let preferred = request
        .headers()
        .get(NODE_HEADER)
        .and_then(|v| v.to_str().ok())
        .filter(|value| validate_node_id(value).is_ok())
        .map(str::to_string);
    let length = match content_length(request.headers()) {
        Ok(length) => length,
        Err(error) => {
            return router_error(
                &trace,
                StatusCode::BAD_REQUEST,
                "invalid_request",
                &error,
                false,
            )
        }
    };
    if length.is_some_and(|n| n > pool.max_request_bytes()) {
        return router_error(
            &trace,
            StatusCode::PAYLOAD_TOO_LARGE,
            "request_too_large",
            "request body is too large",
            false,
        );
    }
    let queued_at = Instant::now();
    trace.state.lock().unwrap().queued_at = Some(queued_at);
    let acquired = pool.acquire(preferred.as_deref(), &trace.id).await;
    {
        let mut state = trace.state.lock().unwrap();
        state.queue_ms = queued_at.elapsed().as_millis();
        state.queued_at = None;
    }
    let mut guard = match acquired {
        Ok(guard) => guard,
        Err(AcquireError::QueueFull) => {
            return router_error(
                &trace,
                StatusCode::SERVICE_UNAVAILABLE,
                "router_busy",
                "router queue is full",
                true,
            )
        }
        Err(AcquireError::TimedOut) => {
            return router_error(
                &trace,
                StatusCode::SERVICE_UNAVAILABLE,
                "no_node",
                "no compatible node has an available slot",
                true,
            )
        }
    };
    trace.state.lock().unwrap().node = Some(guard.node_id.clone());
    let (parts, body) = request.into_parts();
    let mut request_headers = headers::forwarded(
        &parts.headers,
        &[
            "host",
            "expect",
            "content-length",
            NODE_HEADER,
            "x-request-id",
        ],
    );
    request_headers.insert("x-request-id", HeaderValue::from_str(&trace.id).unwrap());
    if let Some(length) = length {
        request_headers.insert(
            http::header::CONTENT_LENGTH,
            HeaderValue::from_str(&length.to_string()).unwrap(),
        );
    }
    if guard
        .send(Payload::RequestHead(RequestHead {
            headers: encode_headers(&request_headers),
            content_length: length,
        }))
        .await
        .is_err()
    {
        guard.cancel_reason = "tunnel_disconnected";
        return tunnel_transport_error(&trace, guard.cancel_reason);
    }
    let slot = guard.slot.clone();
    let upload_trace = trace.clone();
    let limit = pool.max_request_bytes();
    let mut upload = UploadTask(tokio::spawn(async move {
        upload_body(slot, upload_trace, body, length, limit).await
    }));
    let mut uploaded = false;
    // Read responses while uploading: an early rejection must not deadlock either direction.
    let first = loop {
        tokio::select! {
            biased;
            frame = guard.receive() => break frame,
            result = &mut upload.0, if !uploaded => {
                uploaded = true;
                match result {
                    Ok(Ok(())) => {},
                    Ok(Err(error)) => {
                        guard.cancel_reason = error.reason;
                        let response = router_error(&trace, error.status, error.kind, error.message, false);
                        trace.state.lock().unwrap().reason = error.reason;
                        return response;
                    },
                    Err(_) => { guard.cancel_reason = "tunnel_disconnected"; return tunnel_transport_error(&trace, guard.cancel_reason); }
                }
            }
        }
    };
    let response_head = match first {
        Ok(SlotFrame {
            payload: Some(Payload::ResponseHead(head)),
            ..
        }) => head,
        Ok(_) => {
            guard.cancel_reason = "invalid_response";
            return tunnel_transport_error(&trace, guard.cancel_reason);
        }
        Err(error) => {
            guard.cancel_reason = if error.code() == tonic::Code::DataLoss {
                "invalid_response"
            } else {
                "tunnel_disconnected"
            };
            return tunnel_transport_error(&trace, guard.cancel_reason);
        }
    };
    let status = match u16::try_from(response_head.status)
        .ok()
        .and_then(|s| StatusCode::from_u16(s).ok())
        .filter(|s| !s.is_informational())
    {
        Some(status) => status,
        None => {
            guard.cancel_reason = "invalid_response";
            return tunnel_transport_error(&trace, guard.cancel_reason);
        }
    };
    // Let HTTP framing follow EndOfBody. A forwarded Content-Length can cause
    // Hyper to drop the stream before polling the final frame and returning the slot.
    let mut response_headers = match decode_headers(response_head.headers) {
        Ok(h) => headers::forwarded(&h, &[NODE_HEADER, "x-request-id", "content-length"]),
        Err(_) => {
            guard.cancel_reason = "invalid_response";
            return tunnel_transport_error(&trace, guard.cancel_reason);
        }
    };
    response_headers.insert(NODE_HEADER, HeaderValue::from_str(&guard.node_id).unwrap());
    response_headers.insert("x-request-id", HeaderValue::from_str(&trace.id).unwrap());
    trace.state.lock().unwrap().status = status.as_u16();
    if status == StatusCode::SERVICE_UNAVAILABLE {
        pool.mark_saturated(&guard.node_id).await;
    }
    let mut response = HttpResponse::new(Body::from_stream(tunnel_response_stream(
        guard, upload, uploaded, trace,
    )));
    *response.status_mut() = status;
    *response.headers_mut() = response_headers;
    response
}

async fn upload_body(
    slot: Arc<Slot>,
    trace: Arc<RequestTrace>,
    body: Body,
    length: Option<u64>,
    limit: u64,
) -> Result<(), UploadError> {
    let transport = || UploadError {
        status: StatusCode::BAD_GATEWAY,
        kind: "server_transport",
        reason: "tunnel_disconnected",
        message: "tunnel failed; execution outcome is unknown and was not retried",
    };
    let mut body = body.into_data_stream();
    let mut sent = 0_u64;
    let mut closed = slot.subscribe_closed();
    loop {
        let chunk = tokio::select! {
            chunk = body.next() => chunk,
            _ = closed.changed() => return Err(transport()),
        };
        let Some(chunk) = chunk else { break };
        let chunk = chunk.map_err(|_| UploadError {
            status: StatusCode::BAD_REQUEST,
            kind: "invalid_request",
            reason: "client_disconnected",
            message: "client request body failed while streaming",
        })?;
        sent = sent
            .checked_add(chunk.len() as u64)
            .filter(|n| *n <= limit)
            .ok_or(UploadError {
                status: StatusCode::PAYLOAD_TOO_LARGE,
                kind: "request_too_large",
                reason: "request_rejected",
                message: "request body is too large",
            })?;
        for offset in (0..chunk.len()).step_by(DATA_CHUNK_BYTES) {
            let end = (offset + DATA_CHUNK_BYTES).min(chunk.len());
            slot.send(SlotFrame {
                request_id: trace.id.clone(),
                payload: Some(Payload::Data(chunk.slice(offset..end))),
            })
            .await
            .map_err(|_| transport())?;
            trace
                .sent
                .fetch_add((end - offset) as u64, AtomicOrdering::Relaxed);
        }
    }
    if length.is_some_and(|n| n != sent) {
        return Err(UploadError {
            status: StatusCode::BAD_REQUEST,
            kind: "invalid_request",
            reason: "request_rejected",
            message: "Content-Length does not match request body",
        });
    }
    slot.send(SlotFrame {
        request_id: trace.id.clone(),
        payload: Some(Payload::End(EndOfBody {})),
    })
    .await
    .map_err(|_| transport())
}

fn tunnel_response_stream(
    mut guard: SlotGuard,
    mut upload: UploadTask,
    mut uploaded: bool,
    trace: Arc<RequestTrace>,
) -> impl Stream<Item = Result<Bytes, io::Error>> + Send + 'static {
    try_stream! {
        loop {
            let received = tokio::select! {
                biased;
                frame = guard.receive() => frame.map_err(|error| {
                    let reason = if error.code() == tonic::Code::DataLoss {
                        "invalid_response"
                    } else { "tunnel_disconnected" };
                    (error, reason)
                }),
                result = &mut upload.0, if !uploaded => {
                    uploaded = true;
                    match result {
                        Ok(Ok(())) => continue,
                        Ok(Err(error)) => Err((Status::aborted(error.message), error.reason)),
                        Err(_) => Err((Status::unavailable("upload task failed"), "tunnel_disconnected")),
                    }
                }
            };
            let frame = match received {
                Ok(frame) => frame,
                Err((error, reason)) => {
                    guard.cancel_reason = reason;
                    trace.state.lock().unwrap().reason = reason;
                    Err(io::Error::new(io::ErrorKind::ConnectionAborted, error))?
                }
            };
            match frame.payload {
                Some(Payload::Data(data)) => {
                    trace.received.fetch_add(data.len() as u64, AtomicOrdering::Relaxed);
                    yield data;
                },
                Some(Payload::End(_)) => {
                    // Do not wait for a client that stopped uploading after an early response.
                    if !uploaded && upload.0.is_finished() { uploaded = matches!((&mut upload.0).await, Ok(Ok(()))); }
                    trace.state.lock().unwrap().reason = "completed";
                    if uploaded { guard.finish().await; }
                    else { guard.cancel_reason = "request_rejected"; }
                    break;
                },
                _ => {
                    guard.cancel_reason = "invalid_response";
                    trace.state.lock().unwrap().reason = "invalid_response";
                    Err(io::Error::new(io::ErrorKind::InvalidData, "invalid response frame"))?;
                }
            }
        }
    }
}

fn content_length(headers: &HeaderMap) -> Result<Option<u64>, String> {
    match headers.get(http::header::CONTENT_LENGTH) {
        Some(value) => value
            .to_str()
            .ok()
            .and_then(|v| v.parse().ok())
            .map(Some)
            .ok_or_else(|| "invalid Content-Length".to_string()),
        None => Ok(None),
    }
}

fn encode_headers(headers: &HeaderMap) -> Vec<HttpHeader> {
    headers
        .iter()
        .map(|(name, value)| HttpHeader {
            name: name.as_str().as_bytes().to_vec(),
            value: value.as_bytes().to_vec(),
        })
        .collect()
}

fn decode_headers(headers: Vec<HttpHeader>) -> Result<HeaderMap, http::Error> {
    let mut result = HeaderMap::new();
    for header in headers {
        result.append(
            HeaderName::from_bytes(&header.name)?,
            HeaderValue::from_bytes(&header.value)?,
        );
    }
    Ok(result)
}

fn tunnel_transport_error(trace: &RequestTrace, reason: &'static str) -> Response {
    let response = router_error(
        trace,
        StatusCode::BAD_GATEWAY,
        "server_transport",
        "KCoral Server tunnel failed; execution outcome is unknown and was not retried",
        false,
    );
    trace.state.lock().unwrap().reason = reason;
    response
}

fn router_error(
    trace: &RequestTrace,
    status: StatusCode,
    kind: &str,
    message: &str,
    retry_after: bool,
) -> Response {
    trace.finish(
        status.as_u16(),
        match kind {
            "server_transport" => "tunnel_disconnected",
            "router_busy" => "queue_full",
            "no_node" => "queue_timeout",
            _ => "request_rejected",
        },
    );
    let body = Json(
        json!({ "status": "ERROR", "request_id": trace.id, "error": { "kind": kind, "message": message } }),
    );
    let mut response = (status, body).into_response();
    response
        .headers_mut()
        .insert("x-request-id", HeaderValue::from_str(&trace.id).unwrap());
    if retry_after {
        response
            .headers_mut()
            .insert(http::header::RETRY_AFTER, HeaderValue::from_static("1"));
    }
    response
}

pub async fn serve(
    listen: SocketAddr,
    pool: NodePool,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(listen).await?;
    info!(listen = %listener.local_addr()?, "KCoral router listening");
    axum::serve(listener, app(pool))
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preserves_repeated_binary_safe_headers() {
        let mut headers = HeaderMap::new();
        headers.append("set-cookie", HeaderValue::from_static("a=1"));
        headers.append("set-cookie", HeaderValue::from_static("b=2"));
        let decoded = decode_headers(encode_headers(&headers)).unwrap();
        assert_eq!(decoded.get_all("set-cookie").iter().count(), 2);
    }
}
