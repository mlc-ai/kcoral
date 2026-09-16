use std::{
    ffi::OsString,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use axum::{extract::State, response::Json, routing::get, Router};
use bytes::Bytes;
use futures_util::StreamExt;
use http::StatusCode;
use kcoral::{
    proto::{
        router_gateway_client::RouterGatewayClient, slot_frame::Payload, EndOfBody, HttpHeader,
        ResponseHead, SlotFrame, SlotHello, SupervisorStatus,
    },
    router::{self, NodePool, RouterConfig, DATA_CHUNK_BYTES},
    supervisor::{run_server_lifecycle, RouterLinkConfig, ServerLifecycleConfig, SupervisorState},
};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::sync::{mpsc, watch, Mutex, Notify};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{metadata::MetadataValue, Request};

struct RunningRouter {
    pool: NodePool,
    url: String,
    shutdown: watch::Sender<bool>,
    task: tokio::task::JoinHandle<()>,
}

impl RunningRouter {
    async fn stop(self) {
        let _ = self.shutdown.send(true);
        self.task.await.unwrap();
    }
}

async fn start_router(max_queued_requests: usize, token: Option<&str>) -> RunningRouter {
    let pool = NodePool::new(RouterConfig {
        status_check_interval: Duration::from_millis(20),
        max_status_age: Duration::from_secs(5),
        node_retention: Duration::from_secs(60),
        unhealthy_threshold: 3,
        recovery_threshold: 1,
        queue_wait_timeout: Duration::from_millis(250),
        max_queued_requests,
        max_request_bytes: 8 * 1024 * 1024,
        node_token: token.map(str::to_string),
    })
    .unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (shutdown_tx, mut shutdown_rx) = watch::channel(false);
    let app = router::app(pool.clone());
    let task = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                while !*shutdown_rx.borrow() && shutdown_rx.changed().await.is_ok() {}
            })
            .await
            .unwrap();
    });
    RunningRouter {
        pool,
        url: format!("http://{address}"),
        shutdown: shutdown_tx,
        task,
    }
}

#[derive(Default)]
struct FakeNodeState {
    requests: AtomicUsize,
    fail_next: AtomicBool,
    stream_forever: AtomicBool,
    gate: Mutex<Option<Arc<Notify>>>,
    bodies: Mutex<Vec<Vec<u8>>>,
}

struct RunningNode {
    status_tx: mpsc::Sender<SupervisorStatus>,
    state: Arc<FakeNodeState>,
    tasks: Vec<tokio::task::JoinHandle<()>>,
}

impl RunningNode {
    async fn set_healthy(&self, node_id: &str, accepting: bool) {
        self.status_tx
            .send(status(node_id, accepting, 1))
            .await
            .unwrap();
    }

    fn abort(self) {
        for task in self.tasks {
            task.abort();
        }
    }
}

async fn start_node(
    router_url: &str,
    node_id: &str,
    slots: usize,
    token: Option<&str>,
) -> RunningNode {
    let channel = tonic::transport::Endpoint::from_shared(router_url.to_string())
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = RouterGatewayClient::new(channel);
    let (status_tx, status_rx) = mpsc::channel(8);
    status_tx
        .send(status(node_id, true, slots as u64))
        .await
        .unwrap();
    let mut request = Request::new(ReceiverStream::new(status_rx));
    add_token(&mut request, token);
    let mut commands = client
        .connect_supervisor(request)
        .await
        .unwrap()
        .into_inner();
    assert!(commands.message().await.unwrap().is_some());
    let control_task = tokio::spawn(async move {
        while commands
            .message()
            .await
            .is_ok_and(|message| message.is_some())
        {}
    });

    let state = Arc::new(FakeNodeState::default());
    let mut tasks = vec![control_task];
    for slot_index in 0..slots {
        let mut data_client = client.clone();
        let (data_tx, data_rx) = mpsc::channel(2);
        data_tx
            .send(SlotFrame {
                request_id: String::new(),
                payload: Some(Payload::Hello(SlotHello {
                    node_id: node_id.to_string(),
                    server_instance_id: format!("server-{node_id}"),
                    slot_id: format!("{node_id}-{slot_index}"),
                    protocol_version: 1,
                })),
            })
            .await
            .unwrap();
        let mut request = Request::new(ReceiverStream::new(data_rx));
        add_token(&mut request, token);
        let incoming = data_client
            .connect_slot(request)
            .await
            .unwrap()
            .into_inner();
        let task_state = state.clone();
        tasks.push(tokio::spawn(run_fake_slot(incoming, data_tx, task_state)));
    }
    RunningNode {
        status_tx,
        state,
        tasks,
    }
}

fn add_token<T>(request: &mut Request<T>, token: Option<&str>) {
    if let Some(token) = token {
        request.metadata_mut().insert(
            "authorization",
            MetadataValue::try_from(format!("Bearer {token}")).unwrap(),
        );
    }
}

fn status(node_id: &str, healthy: bool, workers: u64) -> SupervisorStatus {
    SupervisorStatus {
        node_id: node_id.to_string(),
        server_healthy: healthy,
        supervisor_instance_id: format!("supervisor-{node_id}"),
        server_instance_id: format!("server-{node_id}"),
        worker_count: workers,
        busy_workers: 0,
        target: [("arch".to_string(), "sm_100a".to_string())].into(),
        versions: [("cuda".to_string(), "13.0".to_string())].into(),
        last_error: String::new(),
        health_age_millis: 0,
    }
}

async fn run_fake_slot(
    mut incoming: tonic::Streaming<SlotFrame>,
    outgoing: mpsc::Sender<SlotFrame>,
    state: Arc<FakeNodeState>,
) {
    let mut request_id = String::new();
    let mut request_headers = Vec::new();
    let mut body = Vec::new();
    while let Some(frame) = incoming.message().await.unwrap() {
        match frame.payload {
            Some(Payload::Ack(_)) => continue,
            Some(Payload::RequestHead(head)) => {
                request_id = frame.request_id;
                request_headers = head.headers;
                body.clear();
            }
            Some(Payload::Data(data)) => body.extend_from_slice(&data),
            Some(Payload::End(_)) => {
                state.requests.fetch_add(1, Ordering::AcqRel);
                state.bodies.lock().await.push(body.clone());
                if state.fail_next.swap(false, Ordering::AcqRel) {
                    return;
                }
                if let Some(gate) = state.gate.lock().await.clone() {
                    gate.notified().await;
                }
                let mut headers = request_headers
                    .iter()
                    .filter(|header| header.name == b"content-type")
                    .cloned()
                    .collect::<Vec<_>>();
                headers.push(HttpHeader {
                    name: b"set-cookie".to_vec(),
                    value: b"a=1".to_vec(),
                });
                headers.push(HttpHeader {
                    name: b"set-cookie".to_vec(),
                    value: b"b=2".to_vec(),
                });
                outgoing
                    .send(SlotFrame {
                        request_id: request_id.clone(),
                        payload: Some(Payload::ResponseHead(ResponseHead {
                            status: 200,
                            headers,
                        })),
                    })
                    .await
                    .unwrap();
                for chunk in body.chunks(DATA_CHUNK_BYTES) {
                    outgoing
                        .send(SlotFrame {
                            request_id: request_id.clone(),
                            payload: Some(Payload::Data(Bytes::copy_from_slice(chunk))),
                        })
                        .await
                        .unwrap();
                }
                if state.stream_forever.load(Ordering::Acquire) {
                    std::future::pending::<()>().await;
                }
                outgoing
                    .send(SlotFrame {
                        request_id: request_id.clone(),
                        payload: Some(Payload::End(EndOfBody {})),
                    })
                    .await
                    .unwrap();
            }
            Some(Payload::Cancel(_)) => return,
            _ => panic!("Router sent an invalid frame to the fake data slot"),
        }
    }
}

async fn wait_ready(router_url: &str) {
    let client = reqwest::Client::new();
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if client
                .get(format!("{router_url}/health"))
                .send()
                .await
                .is_ok_and(|response| response.status() == StatusCode::OK)
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn outbound_only_node_preserves_large_bodies_and_repeated_headers() {
    let router = start_router(8, Some("secret")).await;
    // The fake node opens only outbound gRPC calls and owns no listening socket.
    let node = start_node(&router.url, "gpu-a", 1, Some("secret")).await;
    wait_ready(&router.url).await;
    let body = (0..(2 * 1024 * 1024 + 17))
        .map(|index| (index % 251) as u8)
        .collect::<Vec<_>>();
    let response = reqwest::Client::new()
        .post(format!("{}/execute", router.url))
        .header(
            "content-type",
            "multipart/form-data; boundary=test-boundary",
        )
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["x-kcoral-node"], "gpu-a");
    assert_eq!(response.headers().get_all("set-cookie").iter().count(), 2);
    assert_eq!(response.bytes().await.unwrap(), body);
    assert_eq!(node.state.requests.load(Ordering::Acquire), 1);
    node.abort();
    router.stop().await;
}

#[tokio::test]
async fn control_updates_disable_and_restore_a_node() {
    let router = start_router(8, None).await;
    let node = start_node(&router.url, "gpu-a", 1, None).await;
    wait_ready(&router.url).await;
    for _ in 0..3 {
        node.set_healthy("gpu-a", false).await;
    }
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if router.pool.snapshot().await["workers"][0]["status"] == "unhealthy" {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    node.set_healthy("gpu-a", true).await;
    wait_ready(&router.url).await;
    node.abort();
    router.stop().await;
}

#[tokio::test]
async fn bounds_and_times_out_the_wait_queue() {
    let router = start_router(1, None).await;
    let node = start_node(&router.url, "gpu-a", 1, None).await;
    let gate = Arc::new(Notify::new());
    *node.state.gate.lock().await = Some(gate.clone());
    wait_ready(&router.url).await;
    let client = reqwest::Client::new();
    let first_client = client.clone();
    let first_url = router.url.clone();
    let first = tokio::spawn(async move {
        first_client
            .post(format!("{first_url}/execute"))
            .body("first")
            .send()
            .await
            .unwrap()
    });
    tokio::time::timeout(Duration::from_secs(2), async {
        while node.state.requests.load(Ordering::Acquire) != 1 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();

    let queued_client = client.clone();
    let queued_url = router.url.clone();
    let queued = tokio::spawn(async move {
        queued_client
            .post(format!("{queued_url}/execute"))
            .body("queued")
            .send()
            .await
            .unwrap()
    });
    tokio::time::timeout(Duration::from_secs(2), async {
        while router.pool.snapshot().await["queue_length"] != 1 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    let full = client
        .post(format!("{}/execute", router.url))
        .body("full")
        .send()
        .await
        .unwrap();
    assert_eq!(full.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        full.json::<Value>().await.unwrap()["error"]["kind"],
        "router_busy"
    );
    let timed_out = queued.await.unwrap();
    assert_eq!(timed_out.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        timed_out.json::<Value>().await.unwrap()["error"]["kind"],
        "no_node"
    );
    gate.notify_one();
    assert_eq!(first.await.unwrap().status(), StatusCode::OK);
    node.abort();
    router.stop().await;
}

#[tokio::test]
async fn client_disconnect_discards_the_slot_and_releases_capacity() {
    let router = start_router(0, None).await;
    let node = start_node(&router.url, "gpu-a", 1, None).await;
    node.state.stream_forever.store(true, Ordering::Release);
    wait_ready(&router.url).await;
    let response = reqwest::Client::new()
        .post(format!("{}/execute", router.url))
        .body("first-response-chunk")
        .send()
        .await
        .unwrap();
    let mut stream = response.bytes_stream();
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        "first-response-chunk"
    );
    drop(stream);
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let health = router.pool.snapshot().await;
            if health["active_requests"] == 0 && health["workers"][0]["connected_data_slots"] == 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    node.abort();
    router.stop().await;
}

#[tokio::test]
async fn schedules_without_starving_either_outbound_node() {
    let router = start_router(8, None).await;
    let node_a = start_node(&router.url, "gpu-a", 1, None).await;
    let node_b = start_node(&router.url, "gpu-b", 1, None).await;
    wait_ready(&router.url).await;
    let client = reqwest::Client::new();
    for index in 0..10 {
        let response = client
            .post(format!("{}/execute", router.url))
            .body(index.to_string())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        response.bytes().await.unwrap();
    }
    let count_a = node_a.state.requests.load(Ordering::Acquire);
    let count_b = node_b.state.requests.load(Ordering::Acquire);
    assert_eq!(count_a + count_b, 10);
    // Equal idle loads prefer the least recently selected node, not a random coin flip.
    assert!(count_a >= 4, "gpu-a received only {count_a} requests");
    assert!(count_b >= 4, "gpu-b received only {count_b} requests");
    node_a.abort();
    node_b.abort();
    router.stop().await;
}

#[tokio::test]
async fn tunnel_failure_is_not_replayed_on_another_node() {
    let router = start_router(8, None).await;
    let failing = start_node(&router.url, "gpu-a", 1, None).await;
    wait_ready(&router.url).await;
    let client = reqwest::Client::new();
    let first = client
        .post(format!("{}/execute", router.url))
        .body("first")
        .send()
        .await
        .unwrap();
    let route = first.headers()["x-kcoral-node"].clone();
    assert_eq!(first.bytes().await.unwrap(), "first");

    let healthy = start_node(&router.url, "gpu-b", 1, None).await;
    wait_ready(&router.url).await;
    failing.state.fail_next.store(true, Ordering::Release);
    let failed = client
        .post(format!("{}/execute", router.url))
        .header("x-kcoral-node", route)
        .body("must-not-replay")
        .send()
        .await
        .unwrap();
    assert_eq!(failed.status(), StatusCode::BAD_GATEWAY);
    let error: Value = failed.json().await.unwrap();
    assert_eq!(error["error"]["kind"], "server_transport");
    assert!(error["error"]["message"]
        .as_str()
        .unwrap()
        .contains("not retried"));
    assert_eq!(healthy.state.requests.load(Ordering::Acquire), 0);
    failing.abort();
    healthy.abort();
    router.stop().await;
}

#[tokio::test]
async fn rejects_node_connections_with_the_wrong_token() {
    let router = start_router(1, Some("correct")).await;
    let channel = tonic::transport::Endpoint::from_shared(router.url.clone())
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = RouterGatewayClient::new(channel);
    let (status_tx, status_rx) = mpsc::channel(1);
    status_tx.send(status("gpu-a", true, 1)).await.unwrap();
    let mut request = Request::new(ReceiverStream::new(status_rx));
    add_token(&mut request, Some("wrong"));
    let error = client.connect_supervisor(request).await.unwrap_err();
    assert_eq!(error.code(), tonic::Code::Unauthenticated);
    router.stop().await;
}

#[derive(Clone)]
struct HealthState {
    instance_id: Arc<str>,
}

async fn health(State(state): State<HealthState>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "instance_id": state.instance_id.as_ref(),
        "worker_count": 1,
        "busy_workers": 0,
        "target": {"arch": "sm_100a"},
        "versions": {"cuda": "13.0"},
    }))
}

async fn start_health_server(_healthy: bool) -> (HealthState, reqwest::Url) {
    let state = HealthState {
        instance_id: Arc::from(uuid::Uuid::new_v4().to_string()),
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = Router::new()
        .route("/internal/worker-status", get(health))
        .with_state(state.clone());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (
        state,
        reqwest::Url::parse(&format!("http://{address}/")).unwrap(),
    )
}

#[tokio::test]
async fn supervisor_opens_its_control_connection_outbound() {
    let router = start_router(1, Some("secret")).await;
    let (_health, server_url) = start_health_server(true).await;
    let service =
        SupervisorState::new("gpu-a".to_string(), server_url, Duration::from_millis(100)).unwrap();
    service.refresh_health().await.unwrap();
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let task = tokio::spawn(service.run_status_reporter(
        RouterLinkConfig {
            router_endpoint: router.url.clone(),
            node_token: Some("secret".to_string()),
            heartbeat_interval: Duration::from_millis(20),
            connect_timeout: Duration::from_secs(1),
            reconnect_min_delay: Duration::from_millis(10),
            reconnect_max_delay: Duration::from_millis(40),
        },
        shutdown_rx,
    ));
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if router.pool.snapshot().await["workers"]
                .as_array()
                .unwrap()
                .len()
                == 1
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    shutdown_tx.send(true).unwrap();
    task.await.unwrap().unwrap();
    router.stop().await;
}

#[tokio::test]
async fn supervisor_restarts_an_unhealthy_child_without_a_router() {
    let temporary = TempDir::new().unwrap();
    let starts = temporary.path().join("starts");
    let script = format!("printf x >> '{}'; exec sleep 30", starts.display());
    let command = vec![
        OsString::from("/bin/sh"),
        OsString::from("-c"),
        OsString::from(script),
    ];
    let service = SupervisorState::new(
        "gpu-a".to_string(),
        reqwest::Url::parse("http://127.0.0.1:1/").unwrap(),
        Duration::from_millis(20),
    )
    .unwrap();
    let config = ServerLifecycleConfig {
        health_interval: Duration::from_millis(20),
        failure_threshold: 1,
        startup_grace: Duration::ZERO,
        stable_reset: Duration::from_secs(1),
        termination_grace: Duration::from_millis(50),
        restart_min_delay: Duration::from_millis(10),
        restart_max_delay: Duration::from_millis(40),
        restart_jitter: 0.0,
    };
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let supervisor = tokio::spawn(run_server_lifecycle(
        config,
        service,
        command,
        Vec::new(),
        shutdown_rx,
    ));
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if std::fs::read(&starts).is_ok_and(|contents| contents.len() >= 2) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    shutdown_tx.send(true).unwrap();
    tokio::time::timeout(Duration::from_secs(2), supervisor)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn supervisor_does_not_restart_a_healthy_server() {
    let (_health_state, server_url) = start_health_server(false).await;
    let temporary = TempDir::new().unwrap();
    let starts = temporary.path().join("starts");
    let script = format!("printf x >> '{}'; exec sleep 30", starts.display());
    let command = vec![
        OsString::from("/bin/sh"),
        OsString::from("-c"),
        OsString::from(script),
    ];
    let service =
        SupervisorState::new("gpu-a".to_string(), server_url, Duration::from_millis(100)).unwrap();
    let config = ServerLifecycleConfig {
        health_interval: Duration::from_millis(20),
        failure_threshold: 1,
        startup_grace: Duration::ZERO,
        stable_reset: Duration::from_millis(50),
        termination_grace: Duration::from_millis(50),
        restart_min_delay: Duration::from_millis(10),
        restart_max_delay: Duration::from_millis(40),
        restart_jitter: 0.0,
    };
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let supervisor = tokio::spawn(run_server_lifecycle(
        config,
        service,
        command,
        Vec::new(),
        shutdown_rx,
    ));
    tokio::time::timeout(Duration::from_secs(1), async {
        while !starts.exists() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert_eq!(std::fs::read(&starts).unwrap(), b"x");
    shutdown_tx.send(true).unwrap();
    tokio::time::timeout(Duration::from_secs(2), supervisor)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}
