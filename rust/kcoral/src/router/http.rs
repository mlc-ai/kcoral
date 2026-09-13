use super::{pool::Slot, AcquireError, NodePool, SlotGuard, DATA_CHUNK_BYTES, NODE_HEADER};
use crate::{
    headers,
    proto::{slot_frame::Payload, EndOfBody, HttpHeader, RequestHead, SlotFrame},
    validate_node_id,
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
enum ForwardError {
    InvalidRequest(&'static str),
    ClientBody,
    TooLarge,
    QueueFull,
    QueueTimeout,
    Transport,
    UploadTransport,
    InvalidResponse,
}

impl ForwardError {
    fn reason(&self) -> &'static str {
        match self {
            Self::ClientBody => "client_disconnected",
            Self::QueueFull => "queue_full",
            Self::QueueTimeout => "queue_timeout",
            Self::Transport | Self::UploadTransport => "tunnel_disconnected",
            Self::InvalidResponse => "invalid_response",
            _ => "request_rejected",
        }
    }

    fn response(&self, trace: &RequestTrace) -> Response {
        let (status, kind, message) = match self {
            Self::InvalidRequest(message) => (StatusCode::BAD_REQUEST, "invalid_request", *message),
            Self::ClientBody => (
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "client request body failed while streaming",
            ),
            Self::TooLarge => (
                StatusCode::PAYLOAD_TOO_LARGE,
                "request_too_large",
                "request body is too large",
            ),
            Self::QueueFull => (
                StatusCode::SERVICE_UNAVAILABLE,
                "router_busy",
                "router queue is full",
            ),
            Self::QueueTimeout => (
                StatusCode::SERVICE_UNAVAILABLE,
                "no_node",
                "no compatible node has an available slot",
            ),
            Self::UploadTransport => (
                StatusCode::BAD_GATEWAY,
                "server_transport",
                "tunnel failed; execution outcome is unknown and was not retried",
            ),
            Self::Transport | Self::InvalidResponse => (
                StatusCode::BAD_GATEWAY,
                "server_transport",
                "KCoral Server tunnel failed; execution outcome is unknown and was not retried",
            ),
        };
        trace.finish(status.as_u16(), self.reason());
        let mut response = (
            status,
            Json(json!({
                "status": "ERROR", "request_id": trace.id,
                "error": { "kind": kind, "message": message }
            })),
        )
            .into_response();
        response
            .headers_mut()
            .insert("x-request-id", HeaderValue::from_str(&trace.id).unwrap());
        if matches!(self, Self::QueueFull | Self::QueueTimeout) {
            response
                .headers_mut()
                .insert(http::header::RETRY_AFTER, HeaderValue::from_static("1"));
        }
        response
    }
}

struct UploadTask(tokio::task::JoinHandle<Result<(), ForwardError>>);
impl Drop for UploadTask {
    fn drop(&mut self) {
        self.0.abort();
    }
}

// One owner for the upload and response, including early responses and cancellation.
struct Transfer {
    guard: SlotGuard,
    upload: UploadTask,
    uploaded: bool,
    trace: Arc<RequestTrace>,
}

impl Transfer {
    fn fail(&mut self, error: ForwardError) -> ForwardError {
        self.guard.cancel_reason = error.reason();
        self.trace.state.lock().unwrap().reason = error.reason();
        error
    }

    async fn receive(&mut self) -> Result<SlotFrame, ForwardError> {
        let result = loop {
            tokio::select! {
                biased;
                frame = self.guard.receive() => break frame.map_err(|error| {
                    if error.code() == tonic::Code::DataLoss { ForwardError::InvalidResponse }
                    else { ForwardError::Transport }
                }),
                result = &mut self.upload.0, if !self.uploaded => {
                    self.uploaded = true;
                    if let Err(error) = result.unwrap_or(Err(ForwardError::Transport)) {
                        break Err(error);
                    }
                }
            }
        };
        result.map_err(|error| self.fail(error))
    }

    fn response_body(mut self) -> impl Stream<Item = Result<Bytes, io::Error>> + Send + 'static {
        try_stream! {
            loop {
                let frame = self.receive().await.map_err(|error| {
                    io::Error::new(io::ErrorKind::ConnectionAborted, error.reason())
                })?;
                match frame.payload {
                    Some(Payload::Data(data)) => {
                        self.trace.received.fetch_add(data.len() as u64, AtomicOrdering::Relaxed);
                        yield data;
                    },
                    Some(Payload::End(_)) => {
                        // Never wait for an upload the client stopped after an early response.
                        if !self.uploaded && self.upload.0.is_finished() {
                            self.uploaded = matches!((&mut self.upload.0).await, Ok(Ok(())));
                        }
                        self.trace.state.lock().unwrap().reason = "completed";
                        if self.uploaded { self.guard.finish().await; }
                        else {
                            self.guard.cancel_reason = "request_rejected";
                            drop(self);
                        }
                        break;
                    },
                    _ => {
                        self.fail(ForwardError::InvalidResponse);
                        Err(io::Error::new(io::ErrorKind::InvalidData, "invalid response frame"))?;
                    }
                }
            }
        }
    }
}

async fn execute_handler(State(pool): State<NodePool>, request: HttpRequest<Body>) -> Response {
    let trace = RequestTrace::new();
    match forward_request(pool, request, trace.clone()).await {
        Ok(response) => response,
        Err(error) => error.response(&trace),
    }
}

async fn forward_request(
    pool: NodePool,
    request: HttpRequest<Body>,
    trace: Arc<RequestTrace>,
) -> Result<Response, ForwardError> {
    let preferred = request
        .headers()
        .get(NODE_HEADER)
        .and_then(|v| v.to_str().ok())
        .filter(|value| validate_node_id(value).is_ok());
    let length = content_length(request.headers())?;
    if length.is_some_and(|n| n > pool.max_request_bytes()) {
        return Err(ForwardError::TooLarge);
    }
    let queued_at = Instant::now();
    trace.state.lock().unwrap().queued_at = Some(queued_at);
    let acquired = pool.acquire(preferred, &trace.id).await;
    {
        let mut state = trace.state.lock().unwrap();
        state.queue_ms = queued_at.elapsed().as_millis();
        state.queued_at = None;
    }
    let mut guard = acquired.map_err(|error| match error {
        AcquireError::QueueFull => ForwardError::QueueFull,
        AcquireError::TimedOut => ForwardError::QueueTimeout,
    })?;
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
    guard
        .send(Payload::RequestHead(RequestHead {
            headers: encode_headers(&request_headers),
            content_length: length,
        }))
        .await
        .map_err(|_| {
            guard.cancel_reason = "tunnel_disconnected";
            ForwardError::Transport
        })?;
    let slot = guard.slot.clone();
    let upload_trace = trace.clone();
    let limit = pool.max_request_bytes();
    let upload = UploadTask(tokio::spawn(upload_body(
        slot,
        upload_trace,
        body,
        length,
        limit,
    )));
    let mut transfer = Transfer {
        guard,
        upload,
        uploaded: false,
        trace,
    };
    let head = transfer.receive().await?;
    let (status, mut response_headers) =
        decode_response_head(head).map_err(|error| transfer.fail(error))?;
    response_headers.insert(
        NODE_HEADER,
        HeaderValue::from_str(&transfer.guard.node_id).unwrap(),
    );
    response_headers.insert(
        "x-request-id",
        HeaderValue::from_str(&transfer.trace.id).unwrap(),
    );
    transfer.trace.state.lock().unwrap().status = status.as_u16();
    if status == StatusCode::SERVICE_UNAVAILABLE {
        pool.mark_saturated(&transfer.guard.node_id).await;
    }
    let mut response = HttpResponse::new(Body::from_stream(transfer.response_body()));
    *response.status_mut() = status;
    *response.headers_mut() = response_headers;
    Ok(response)
}

fn decode_response_head(frame: SlotFrame) -> Result<(StatusCode, HeaderMap), ForwardError> {
    let Some(Payload::ResponseHead(head)) = frame.payload else {
        return Err(ForwardError::InvalidResponse);
    };
    let status = u16::try_from(head.status)
        .ok()
        .and_then(|s| StatusCode::from_u16(s).ok())
        .filter(|s| !s.is_informational())
        .ok_or(ForwardError::InvalidResponse)?;
    let headers = decode_headers(head.headers).map_err(|_| ForwardError::InvalidResponse)?;
    // HTTP framing must consume EndOfBody before returning the slot to the pool.
    Ok((
        status,
        headers::forwarded(&headers, &[NODE_HEADER, "x-request-id", "content-length"]),
    ))
}

async fn upload_body(
    slot: Arc<Slot>,
    trace: Arc<RequestTrace>,
    body: Body,
    length: Option<u64>,
    limit: u64,
) -> Result<(), ForwardError> {
    let mut body = body.into_data_stream();
    let mut sent = 0_u64;
    let mut closed = slot.subscribe_closed();
    loop {
        let chunk = tokio::select! {
            chunk = body.next() => chunk,
            _ = closed.changed() => return Err(ForwardError::UploadTransport),
        };
        let Some(chunk) = chunk else { break };
        let chunk = chunk.map_err(|_| ForwardError::ClientBody)?;
        sent = sent
            .checked_add(chunk.len() as u64)
            .filter(|n| *n <= limit)
            .ok_or(ForwardError::TooLarge)?;
        for offset in (0..chunk.len()).step_by(DATA_CHUNK_BYTES) {
            let end = (offset + DATA_CHUNK_BYTES).min(chunk.len());
            slot.send(SlotFrame {
                request_id: trace.id.clone(),
                payload: Some(Payload::Data(chunk.slice(offset..end))),
            })
            .await
            .map_err(|_| ForwardError::UploadTransport)?;
            trace
                .sent
                .fetch_add((end - offset) as u64, AtomicOrdering::Relaxed);
        }
    }
    if length.is_some_and(|n| n != sent) {
        return Err(ForwardError::InvalidRequest(
            "Content-Length does not match request body",
        ));
    }
    slot.send(SlotFrame {
        request_id: trace.id.clone(),
        payload: Some(Payload::End(EndOfBody {})),
    })
    .await
    .map_err(|_| ForwardError::UploadTransport)
}

fn content_length(headers: &HeaderMap) -> Result<Option<u64>, ForwardError> {
    match headers.get(http::header::CONTENT_LENGTH) {
        Some(value) => value
            .to_str()
            .ok()
            .and_then(|v| v.parse().ok())
            .map(Some)
            .ok_or(ForwardError::InvalidRequest("invalid Content-Length")),
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
