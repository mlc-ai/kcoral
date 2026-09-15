use super::DATA_CHUNK_BYTES;
use crate::proto::{slot_frame::Payload, Cancel, SlotFrame, SupervisorStatus};
use crate::validate_node_id;
use rand::Rng;
use serde_json::{json, Value};
use std::{
    cmp::Ordering,
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, Mutex, Notify, Semaphore};
use tonic::Status;
use tracing::info;
use uuid::Uuid;
#[derive(Clone, Debug)]
pub struct RouterConfig {
    pub status_check_interval: Duration,
    pub max_status_age: Duration,
    pub node_retention: Duration,
    pub unhealthy_threshold: u32,
    pub recovery_threshold: u32,
    pub queue_wait_timeout: Duration,
    pub max_queued_requests: usize,
    pub max_request_bytes: u64,
    pub node_token: Option<String>,
}

impl RouterConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.status_check_interval.is_zero() || self.max_status_age.is_zero() {
            anyhow::bail!("status intervals must be positive");
        }
        if self.unhealthy_threshold == 0 || self.recovery_threshold == 0 {
            anyhow::bail!("health thresholds must be positive");
        }
        if self.node_retention <= self.max_status_age {
            anyhow::bail!("node retention must exceed maximum status age");
        }
        if self.max_request_bytes == 0 {
            anyhow::bail!("max request bytes must be positive");
        }
        if self.node_token.as_deref() == Some("") {
            anyhow::bail!("node token cannot be empty");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum NodeStatus {
    #[default]
    Starting,
    Ready,
    Recovering,
    Incompatible,
    Unhealthy,
}

impl NodeStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Starting => "starting",
            Self::Ready => "ready",
            Self::Recovering => "recovering",
            Self::Incompatible => "incompatible",
            Self::Unhealthy => "unhealthy",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Cohort {
    target: HashMap<String, String>,
    versions: HashMap<String, String>,
}

pub(super) struct Slot {
    node_id: String,
    server_instance_id: String,
    slot_id: String,
    // Identifies this connection, so a late old cleanup cannot remove a replacement.
    connection_id: String,
    closed: tokio::sync::watch::Sender<Option<String>>,
    outgoing: mpsc::Sender<Result<SlotFrame, Status>>,
    incoming: Mutex<mpsc::Receiver<SlotFrame>>,
}

impl Slot {
    pub(super) fn new(
        node_id: String,
        server_instance_id: String,
        slot_id: String,
        connection_id: String,
        outgoing: mpsc::Sender<Result<SlotFrame, Status>>,
        incoming: mpsc::Receiver<SlotFrame>,
    ) -> Self {
        Self {
            node_id,
            server_instance_id,
            slot_id,
            connection_id,
            outgoing,
            incoming: Mutex::new(incoming),
            closed: tokio::sync::watch::channel(None).0,
        }
    }
    pub(super) fn subscribe_closed(&self) -> tokio::sync::watch::Receiver<Option<String>> {
        self.closed.subscribe()
    }

    fn disconnect(&self, reason: &str) {
        self.closed.send_replace(Some(reason.to_string()));
    }

    pub(super) fn close_status(&self) -> Status {
        Status::cancelled(self.closed.borrow().as_deref().unwrap_or_default())
    }

    pub(super) async fn send(&self, frame: SlotFrame) -> Result<(), Status> {
        let mut closed = self.closed.subscribe();
        if closed.borrow().is_some() {
            return Err(Status::unavailable("slot disconnected"));
        }
        tokio::select! {
            result = self.outgoing.send(Ok(frame)) => result.map_err(|_| Status::unavailable("slot closed")),
            _ = closed.changed() => Err(Status::unavailable("slot disconnected")),
        }
    }

    pub(super) async fn receive(&self) -> Option<SlotFrame> {
        let mut incoming = self.incoming.lock().await;
        let mut closed = self.closed.subscribe();
        if closed.borrow().is_some() {
            return incoming.try_recv().ok();
        }
        tokio::select! {
            biased;
            frame = incoming.recv() => frame,
            _ = closed.changed() => incoming.try_recv().ok(),
        }
    }
}

#[derive(Default)]
struct Node {
    name: String,
    status: NodeStatus,
    control_connection_id: Option<String>,
    supervisor_instance_id: String,
    server_instance_id: String,
    restarts: u64,
    consecutive_failures: u32,
    recovery_successes: u32,
    last_control: Option<Instant>,
    disconnected_since: Option<Instant>,
    last_success: Option<Instant>,
    last_error: String,

    worker_count: u64,
    busy_workers: u64,

    in_flight: u64,
    last_selected: u64,
    saturated_until: Option<Instant>,
    slots: HashMap<String, Arc<Slot>>,
    idle_slots: VecDeque<String>,
}

impl Node {
    fn new(name: String) -> Self {
        Self {
            name,
            disconnected_since: Some(Instant::now()),
            ..Self::default()
        }
    }

    fn estimated_used(&self) -> u64 {
        self.in_flight.max(self.busy_workers)
    }

    fn available_capacity(&self, now: Instant) -> u64 {
        if self.status != NodeStatus::Ready
            || self.control_connection_id.is_none()
            || self.saturated_until.is_some_and(|deadline| deadline > now)
        {
            return 0;
        }
        self.idle_slots.len().try_into().unwrap_or(u64::MAX)
    }
}

#[derive(Default)]
struct PoolState {
    nodes: Vec<Node>,
    node_indices: HashMap<String, usize>,
    cohort: Option<Cohort>,
    selection_sequence: u64,
}

struct PoolInner {
    config: RouterConfig,
    instance_id: String,
    started_at: String,
    state: Mutex<PoolState>,
    notify: Notify,
    queue_slots: Arc<Semaphore>,
}

#[derive(Clone)]
pub struct NodePool {
    inner: Arc<PoolInner>,
}

impl NodePool {
    pub(super) fn max_request_bytes(&self) -> u64 {
        self.inner.config.max_request_bytes
    }

    pub fn new(config: RouterConfig) -> anyhow::Result<Self> {
        config.validate()?;
        Ok(Self {
            inner: Arc::new(PoolInner {
                queue_slots: Arc::new(Semaphore::new(config.max_queued_requests)),
                config,
                instance_id: Uuid::new_v4().to_string(),
                started_at: chrono::DateTime::<chrono::Utc>::from(std::time::SystemTime::now())
                    .to_rfc3339_opts(chrono::SecondsFormat::Micros, true),
                state: Mutex::new(PoolState::default()),
                notify: Notify::new(),
            }),
        })
    }

    pub(super) fn node_token(&self) -> Option<&str> {
        self.inner.config.node_token.as_deref()
    }

    pub(super) async fn record_status(
        &self,
        expected_node_id: Option<&str>,
        connection_id: &str,
        status: SupervisorStatus,
    ) -> Result<String, Status> {
        validate_node_id(&status.node_id)
            .map_err(|error| Status::invalid_argument(error.to_string()))?;
        if expected_node_id.is_some_and(|expected| expected != status.node_id) {
            return Err(Status::invalid_argument(
                "control stream changed its node identifier",
            ));
        }
        if status.supervisor_instance_id.is_empty() {
            return Err(Status::invalid_argument(
                "status is missing the supervisor instance identifier",
            ));
        }

        let node_id = status.node_id.clone();
        let mut state = self.inner.state.lock().await;
        if expected_node_id.is_some() {
            let current = state
                .node_indices
                .get(&node_id)
                .and_then(|index| state.nodes[*index].control_connection_id.as_deref());
            if current != Some(connection_id) {
                return Err(Status::failed_precondition(
                    "supervisor connection was superseded or expired",
                ));
            }
        }
        if status.server_healthy {
            validate_report(&status, self.inner.config.max_status_age)
                .map_err(|error| Status::invalid_argument(error.to_string()))?;
        }
        let index = ensure_node(&mut state, &node_id);
        let replacing_connection = state.nodes[index]
            .control_connection_id
            .as_deref()
            .is_some_and(|current| current != connection_id);
        state.nodes[index].control_connection_id = Some(connection_id.to_string());
        state.nodes[index].last_control = Some(Instant::now());
        state.nodes[index].disconnected_since = None;

        if status.server_healthy {
            apply_success(&self.inner.config, &mut state, index, status);
        } else {
            let node = &mut state.nodes[index];
            if node.supervisor_instance_id != status.supervisor_instance_id
                && !node.supervisor_instance_id.is_empty()
            {
                node.restarts = node.restarts.saturating_add(1);
                disconnect_all_slots(node, "supervisor instance changed");
            }
            node.supervisor_instance_id = status.supervisor_instance_id;
            apply_failure(
                &self.inner.config,
                node,
                if status.last_error.is_empty() {
                    "node reports that the local KCoral Server is unhealthy"
                } else {
                    &status.last_error
                },
            );
        }
        if replacing_connection {
            info!(
                node = node_id,
                "replaced stale supervisor control connection"
            );
        }
        drop(state);
        self.inner.notify.notify_waiters();
        Ok(node_id)
    }

    pub(super) async fn control_disconnected(
        &self,
        node_id: &str,
        connection_id: &str,
        error: &str,
    ) {
        let mut state = self.inner.state.lock().await;
        let Some(index) = state.node_indices.get(node_id).copied() else {
            return;
        };
        let node = &mut state.nodes[index];
        if node.control_connection_id.as_deref() != Some(connection_id) {
            return;
        }
        node.control_connection_id = None;
        node.disconnected_since = Some(Instant::now());
        node.status = NodeStatus::Unhealthy;
        node.consecutive_failures = self.inner.config.unhealthy_threshold;
        node.recovery_successes = 0;
        node.last_error = error.to_string();
        drop(state);
        self.inner.notify.notify_waiters();
    }

    pub(super) async fn register_slot(&self, slot: Arc<Slot>) -> Result<(), Status> {
        validate_node_id(&slot.node_id)
            .map_err(|error| Status::invalid_argument(error.to_string()))?;
        if slot.server_instance_id.is_empty() || slot.slot_id.is_empty() {
            return Err(Status::invalid_argument(
                "data hello is missing an instance or slot identifier",
            ));
        }
        let mut state = self.inner.state.lock().await;
        let index = ensure_node(&mut state, &slot.node_id);
        let node = &mut state.nodes[index];
        if !node.server_instance_id.is_empty() && node.server_instance_id != slot.server_instance_id
        {
            return Err(Status::failed_precondition(
                "data slot does not match the current server instance",
            ));
        }
        if node.slots.contains_key(&slot.slot_id) {
            return Err(Status::already_exists(
                "data slot identifier is already connected",
            ));
        }
        if node.server_instance_id.is_empty() {
            node.server_instance_id = slot.server_instance_id.clone();
        }
        node.idle_slots.push_back(slot.slot_id.clone());
        node.slots.insert(slot.slot_id.clone(), slot.clone());
        info!(
            node = slot.node_id,
            slot = slot.slot_id,
            "registered outbound data slot"
        );
        drop(state);
        self.inner.notify.notify_waiters();
        Ok(())
    }

    pub(super) async fn remove_slot(
        &self,
        node_id: &str,
        slot_id: &str,
        connection_id: &str,
        error: &str,
    ) {
        let mut state = self.inner.state.lock().await;
        let Some(index) = state.node_indices.get(node_id).copied() else {
            return;
        };
        let node = &mut state.nodes[index];
        let matches = node
            .slots
            .get(slot_id)
            .is_some_and(|slot| slot.connection_id == connection_id);
        if !matches {
            return;
        }
        if let Some(slot) = node.slots.remove(slot_id) {
            slot.disconnect(error);
        }
        node.idle_slots.retain(|candidate| candidate != slot_id);
        node.last_error = error.to_string();
        drop(state);
        self.inner.notify.notify_waiters();
    }

    async fn abort_slot(&self, slot: Arc<Slot>, request_id: String, reason: &'static str) {
        // Never wait for remote consumption before returning local capacity.
        let _ = slot.outgoing.try_send(Ok(SlotFrame {
            request_id,
            payload: Some(Payload::Cancel(Cancel {
                reason: reason.to_string(),
            })),
        }));
        self.remove_slot(&slot.node_id, &slot.slot_id, &slot.connection_id, reason)
            .await;
        slot.disconnect(reason);
        let mut state = self.inner.state.lock().await;
        if let Some(index) = state.node_indices.get(&slot.node_id).copied() {
            state.nodes[index].in_flight = state.nodes[index].in_flight.saturating_sub(1);
        }
        drop(state);
        self.inner.notify.notify_waiters();
    }

    async fn try_acquire(&self, preferred: Option<&str>, request_id: &str) -> Option<SlotGuard> {
        let mut state = self.inner.state.lock().await;
        let now = Instant::now();
        let candidates = state
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.available_capacity(now) > 0)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return None;
        }

        let preferred_index = preferred.and_then(|token| {
            candidates
                .iter()
                .copied()
                .find(|index| state.nodes[*index].name == token)
        });
        let selected = preferred_index.unwrap_or_else(|| choose_power_of_two(&state, &candidates));
        state.selection_sequence = state.selection_sequence.wrapping_add(1);
        let sequence = state.selection_sequence;
        let node = &mut state.nodes[selected];
        let slot_id = node.idle_slots.pop_front()?;
        let slot = node.slots.get(&slot_id)?.clone();
        node.in_flight = node.in_flight.saturating_add(1);
        node.last_selected = sequence;
        Some(SlotGuard {
            pool: self.clone(),
            node_id: node.name.clone(),
            request_id: request_id.to_string(),
            slot,
            active: true,
            cancel_reason: "client_disconnected",
        })
    }

    pub(super) async fn acquire(
        &self,
        preferred: Option<&str>,
        request_id: &str,
    ) -> Result<SlotGuard, AcquireError> {
        if let Some(route) = self.try_acquire(preferred, request_id).await {
            return Ok(route);
        }
        let _permit = self
            .inner
            .queue_slots
            .clone()
            .try_acquire_owned()
            .map_err(|_| AcquireError::QueueFull)?;
        let deadline = tokio::time::Instant::now() + self.inner.config.queue_wait_timeout;
        loop {
            let notified = self.inner.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Some(route) = self.try_acquire(preferred, request_id).await {
                return Ok(route);
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                return Err(AcquireError::TimedOut);
            }
        }
    }

    async fn release_slot(&self, node_id: &str, slot: &Arc<Slot>) {
        let mut state = self.inner.state.lock().await;
        let Some(index) = state.node_indices.get(node_id).copied() else {
            return;
        };
        let node = &mut state.nodes[index];
        node.in_flight = node.in_flight.saturating_sub(1);
        let still_registered = node
            .slots
            .get(&slot.slot_id)
            .is_some_and(|registered| registered.connection_id == slot.connection_id);
        if still_registered
            && slot.closed.borrow().is_none()
            && node.server_instance_id == slot.server_instance_id
        {
            node.idle_slots.push_back(slot.slot_id.clone());
        }
        drop(state);
        self.inner.notify.notify_one();
    }

    pub(super) async fn mark_saturated(&self, node_id: &str) {
        let mut state = self.inner.state.lock().await;
        if let Some(index) = state.node_indices.get(node_id).copied() {
            state.nodes[index].saturated_until =
                Some(Instant::now() + self.inner.config.status_check_interval);
        }
        drop(state);
        self.inner.notify.notify_waiters();
    }

    pub async fn run_status_loop(self, mut shutdown: tokio::sync::watch::Receiver<bool>) {
        let mut interval = tokio::time::interval(self.inner.config.status_check_interval);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                _ = interval.tick() => self.expire_stale_nodes().await,
                changed = shutdown.changed() => {
                    if changed.is_err() || *shutdown.borrow() {
                        return;
                    }
                }
            }
        }
    }

    async fn expire_stale_nodes(&self) {
        let mut state = self.inner.state.lock().await;
        let now = Instant::now();
        for node in &mut state.nodes {
            if node.control_connection_id.is_some()
                && node
                    .last_control
                    .is_some_and(|last| now.duration_since(last) > self.inner.config.max_status_age)
            {
                node.status = NodeStatus::Unhealthy;
                node.control_connection_id = None;
                node.disconnected_since = Some(now);
                node.consecutive_failures = self.inner.config.unhealthy_threshold;
                node.recovery_successes = 0;
                node.last_error = "supervisor status expired".to_string();
            }
        }
        state.nodes.retain_mut(|node| {
            let expired = node.control_connection_id.is_none()
                && node.in_flight == 0
                && node.disconnected_since.is_some_and(|since| {
                    now.duration_since(since) >= self.inner.config.node_retention
                });
            if expired {
                disconnect_all_slots(node, "node retention expired");
            }
            !expired
        });
        state.node_indices = state
            .nodes
            .iter()
            .enumerate()
            .map(|(i, node)| (node.name.clone(), i))
            .collect();
        if state.nodes.is_empty() {
            state.cohort = None;
        }
        drop(state);
        self.inner.notify.notify_waiters();
    }

    pub async fn health(&self) -> Value {
        let state = self.inner.state.lock().await;
        let now = Instant::now();
        let ready = state
            .nodes
            .iter()
            .any(|node| node.status == NodeStatus::Ready && !node.slots.is_empty());
        let capacity = state
            .nodes
            .iter()
            .filter(|node| {
                node.status == NodeStatus::Ready
                    && node.control_connection_id.is_some()
                    && !node.saturated_until.is_some_and(|deadline| deadline > now)
            })
            .map(|node| node.slots.len() as u64)
            .sum::<u64>();
        let (target, versions) = state
            .cohort
            .as_ref()
            .map(|cohort| (cohort.target.clone(), cohort.versions.clone()))
            .unwrap_or_default();
        json!({
            "status": if ready { "ok" } else { "unavailable" },
            "instance_id": self.inner.instance_id,
            "started_at": self.inner.started_at,
            "gpu_count": null,
            "load": {
                "request_capacity": capacity,
                "requests_in_progress": state.nodes.iter().map(|node| node.in_flight).sum::<u64>(),
                "requests_waiting": self.inner.config.max_queued_requests - self.inner.queue_slots.available_permits(),
            },
            "target": target,
            "versions": versions,
        })
    }

    pub async fn snapshot(&self) -> Value {
        let state = self.inner.state.lock().await;
        let now = Instant::now();
        let ready = state
            .nodes
            .iter()
            .any(|node| node.status == NodeStatus::Ready && !node.slots.is_empty());
        let (target, versions) = state
            .cohort
            .as_ref()
            .map(|cohort| (cohort.target.clone(), cohort.versions.clone()))
            .unwrap_or_default();
        json!({
            "status": if ready { "ok" } else { "unavailable" },
            "instance_id": self.inner.instance_id,
            "active_requests": state.nodes.iter().map(|node| node.in_flight).sum::<u64>(),
            "queue_length": self.inner.config.max_queued_requests - self.inner.queue_slots.available_permits(),
            "target": target,
            "versions": versions,
            "workers": state.nodes.iter().map(|node| json!({
                "name": node.name,
                "status": node.status.as_str(),
                "supervisor_instance_id": nullable_string(&node.supervisor_instance_id),
                "server_instance_id": nullable_string(&node.server_instance_id),
                "restarts": node.restarts,
                "in_flight": node.in_flight,
                "capacity": node.worker_count,
                "connected_data_slots": node.slots.len(),
                "available_capacity": node.available_capacity(now),
                "consecutive_failures": node.consecutive_failures,
                "recovery_successes": node.recovery_successes,
                "last_probe_ok": node.last_success.is_some_and(|last| now.duration_since(last) <= self.inner.config.max_status_age),
                "last_success_age_seconds": node.last_success.map(|last| now.duration_since(last).as_secs_f64()),
                "last_error": nullable_string(&node.last_error),
            })).collect::<Vec<_>>(),
        })
    }
}

fn ensure_node(state: &mut PoolState, node_id: &str) -> usize {
    if let Some(index) = state.node_indices.get(node_id) {
        return *index;
    }
    let index = state.nodes.len();
    state.nodes.push(Node::new(node_id.to_string()));
    state.node_indices.insert(node_id.to_string(), index);
    index
}

fn disconnect_all_slots(node: &mut Node, reason: &str) {
    for slot in node.slots.values() {
        slot.disconnect(reason);
        let _ = slot
            .outgoing
            .try_send(Err(Status::unavailable(reason.to_string())));
    }
    node.slots.clear();
    node.idle_slots.clear();
}

fn nullable_string(value: &str) -> Option<&str> {
    (!value.is_empty()).then_some(value)
}

fn choose_power_of_two(state: &PoolState, candidates: &[usize]) -> usize {
    if candidates.len() == 1 {
        return candidates[0];
    }
    let mut rng = rand::rng();
    let first_position = rng.random_range(0..candidates.len());
    let second_position =
        (first_position + 1 + rng.random_range(0..candidates.len() - 1)) % candidates.len();
    let first = candidates[first_position];
    let second = candidates[second_position];
    let first_node = &state.nodes[first];
    let second_node = &state.nodes[second];
    let order = compare_load(first_node, second_node)
        .then_with(|| first_node.last_selected.cmp(&second_node.last_selected));
    if order.is_gt() {
        second
    } else {
        first
    }
}

fn compare_load(first: &Node, second: &Node) -> Ordering {
    let first_capacity = first.worker_count.max(1) as u128;
    let second_capacity = second.worker_count.max(1) as u128;
    ((first.estimated_used() as u128) * second_capacity)
        .cmp(&((second.estimated_used() as u128) * first_capacity))
}

fn validate_report(response: &SupervisorStatus, max_health_age: Duration) -> anyhow::Result<()> {
    if response.server_instance_id.is_empty() {
        anyhow::bail!("status is missing the server instance identifier");
    }
    if response.worker_count == 0 || response.busy_workers > response.worker_count {
        anyhow::bail!("status has invalid worker capacity");
    }
    let max_health_age_millis = max_health_age.as_millis().try_into().unwrap_or(u64::MAX);
    if response.health_age_millis > max_health_age_millis {
        anyhow::bail!("node reports a stale local health snapshot");
    }
    Ok(())
}

fn apply_success(
    config: &RouterConfig,
    state: &mut PoolState,
    index: usize,
    value: SupervisorStatus,
) {
    let cohort = Cohort {
        target: value.target,
        versions: value.versions,
    };
    let compatible = state.cohort.get_or_insert_with(|| cohort.clone()) == &cohort;
    let node = &mut state.nodes[index];
    let instance_changed = (!node.server_instance_id.is_empty()
        && node.server_instance_id != value.server_instance_id)
        || (!node.supervisor_instance_id.is_empty()
            && node.supervisor_instance_id != value.supervisor_instance_id);
    if instance_changed {
        node.restarts = node.restarts.saturating_add(1);
        disconnect_all_slots(node, "KCoral Server instance changed");
    }
    node.supervisor_instance_id = value.supervisor_instance_id;
    node.server_instance_id = value.server_instance_id;

    node.worker_count = value.worker_count;
    node.busy_workers = value.busy_workers;

    node.last_success = Some(Instant::now());
    node.last_error.clear();
    node.consecutive_failures = 0;
    node.saturated_until = None;

    if !compatible {
        node.status = NodeStatus::Incompatible;
        node.recovery_successes = 0;
    } else if matches!(node.status, NodeStatus::Unhealthy | NodeStatus::Recovering) {
        node.recovery_successes = node.recovery_successes.saturating_add(1);
        if node.recovery_successes >= config.recovery_threshold {
            node.status = NodeStatus::Ready;
            node.recovery_successes = 0;
        } else {
            node.status = NodeStatus::Recovering;
        }
    } else {
        node.status = NodeStatus::Ready;
        node.recovery_successes = 0;
    }
}

fn apply_failure(config: &RouterConfig, node: &mut Node, error: &str) {
    node.last_error = error.to_string();
    node.recovery_successes = 0;
    node.consecutive_failures = node.consecutive_failures.saturating_add(1);
    if node.consecutive_failures >= config.unhealthy_threshold
        || matches!(node.status, NodeStatus::Starting | NodeStatus::Recovering)
    {
        node.status = NodeStatus::Unhealthy;
    }
}

#[derive(Debug, thiserror::Error)]
pub(super) enum AcquireError {
    #[error("router queue is full")]
    QueueFull,
    #[error("no compatible node became available before the queue timeout")]
    TimedOut,
}

pub(super) struct SlotGuard {
    pool: NodePool,
    pub(super) node_id: String,
    request_id: String,
    pub(super) slot: Arc<Slot>,
    active: bool,
    pub(super) cancel_reason: &'static str,
}

impl SlotGuard {
    pub(super) async fn send(&self, payload: Payload) -> Result<(), Status> {
        self.slot
            .send(SlotFrame {
                request_id: self.request_id.clone(),
                payload: Some(payload),
            })
            .await
    }

    pub(super) async fn receive(&self) -> Result<SlotFrame, Status> {
        let frame = self
            .slot
            .receive()
            .await
            .ok_or_else(|| Status::unavailable("data slot request stream closed"))?;
        if frame.request_id != self.request_id {
            return Err(Status::data_loss(
                "data slot returned the wrong request identifier",
            ));
        }
        if matches!(&frame.payload, Some(Payload::Data(data)) if data.len() > DATA_CHUNK_BYTES) {
            return Err(Status::data_loss(
                "response data frame exceeds the size limit",
            ));
        }
        Ok(frame)
    }

    pub(super) async fn finish(mut self) {
        self.pool.release_slot(&self.node_id, &self.slot).await;
        self.active = false;
    }
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;
        let pool = self.pool.clone();
        let slot = self.slot.clone();
        let request_id = self.request_id.clone();
        let reason = self.cancel_reason;
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                pool.abort_slot(slot, request_id, reason).await;
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> RouterConfig {
        RouterConfig {
            status_check_interval: Duration::from_secs(1),
            max_status_age: Duration::from_secs(5),
            node_retention: Duration::from_secs(60),
            unhealthy_threshold: 3,
            recovery_threshold: 2,
            queue_wait_timeout: Duration::from_secs(1),
            max_queued_requests: 1,
            max_request_bytes: 1024,
            node_token: None,
        }
    }

    fn node(capacity: u64, active: u64, selected: u64) -> Node {
        let mut node = Node::new("node".to_string());
        node.worker_count = capacity;
        node.busy_workers = active;
        node.last_selected = selected;
        node
    }

    fn status(instance: &str, _accepting: bool, arch: &str) -> SupervisorStatus {
        SupervisorStatus {
            node_id: "node".to_string(),
            server_healthy: true,
            supervisor_instance_id: format!("supervisor-{instance}"),
            server_instance_id: format!("server-{instance}"),

            worker_count: 1,
            busy_workers: 0,
            target: [("arch".to_string(), arch.to_string())].into(),
            versions: [("cuda".to_string(), "13.0".to_string())].into(),
            last_error: String::new(),
            health_age_millis: 0,
        }
    }

    #[tokio::test]
    async fn public_health_tracks_capacity_queueing_and_disconnection() {
        let pool = NodePool::new(config()).unwrap();
        pool.record_status(None, "control", status("one", true, "cpu"))
            .await
            .unwrap();
        assert_eq!(pool.health().await["load"]["request_capacity"], 0);
        pool.register_slot(slot("one", "first")).await.unwrap();
        let guard = pool.acquire(None, "assigned").await.unwrap();
        let queued_pool = pool.clone();
        let queued = tokio::spawn(async move { queued_pool.acquire(None, "queued").await });
        tokio::time::timeout(Duration::from_secs(1), async {
            while pool.health().await["load"]["requests_waiting"] != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(
            pool.health().await["load"],
            json!({
                "request_capacity": 1, "requests_in_progress": 1, "requests_waiting": 1
            })
        );
        queued.abort();
        let _ = queued.await;
        pool.control_disconnected("node", "control", "lost control connection")
            .await;
        let disconnected = pool.health().await;
        assert_eq!(disconnected["status"], "unavailable");
        assert_eq!(
            disconnected["load"],
            json!({
                "request_capacity": 0, "requests_in_progress": 1, "requests_waiting": 0
            })
        );
        guard.finish().await;
        assert_eq!(pool.health().await["load"]["requests_in_progress"], 0);
    }

    #[tokio::test]
    async fn public_capacity_excludes_backoff_and_incompatible_nodes() {
        let pool = NodePool::new(config()).unwrap();
        pool.record_status(None, "control", status("one", true, "cpu"))
            .await
            .unwrap();
        pool.register_slot(slot("one", "first")).await.unwrap();
        {
            let mut state = pool.inner.state.lock().await;
            state.nodes[0].saturated_until = Some(Instant::now() + Duration::from_secs(10));
        }
        assert_eq!(pool.health().await["load"]["request_capacity"], 0);
        pool.record_status(Some("node"), "control", status("one", true, "cpu"))
            .await
            .unwrap();
        assert_eq!(pool.health().await["load"]["request_capacity"], 1);
        pool.record_status(Some("node"), "control", status("one", true, "other"))
            .await
            .unwrap();
        assert_eq!(pool.health().await["load"]["request_capacity"], 0);
    }

    #[test]
    fn compares_normalized_load() {
        assert_eq!(
            compare_load(&node(8, 4, 0), &node(2, 0, 0)),
            Ordering::Greater
        );
        assert_eq!(
            compare_load(&node(8, 4, 0), &node(2, 1, 0)),
            Ordering::Equal
        );
        let mut local = node(8, 2, 0);
        local.in_flight = 4;
        assert_eq!(local.estimated_used(), 4);
        assert_eq!(compare_load(&local, &node(2, 1, 0)), Ordering::Equal);
    }

    #[tokio::test]
    async fn health_thresholds_avoid_flapping() {
        let pool = NodePool::new(config()).unwrap();
        pool.record_status(None, "one", status("one", true, "sm_100a"))
            .await
            .unwrap();
        pool.control_disconnected("node", "one", "lost").await;
        pool.record_status(None, "two", status("one", true, "sm_100a"))
            .await
            .unwrap();
        let snapshot = pool.snapshot().await;
        assert_eq!(snapshot["workers"][0]["status"], "recovering");
        pool.record_status(Some("node"), "two", status("one", true, "sm_100a"))
            .await
            .unwrap();
        let snapshot = pool.snapshot().await;
        assert_eq!(snapshot["workers"][0]["status"], "ready");
    }

    #[test]
    fn rejects_stale_status_snapshots() {
        let mut response = status("one", true, "sm_100a");
        response.health_age_millis = 6_000;
        let error = validate_report(&response, Duration::from_secs(5)).unwrap_err();
        assert!(error.to_string().contains("stale"));
    }

    fn slot(instance: &str, connection: &str) -> Arc<Slot> {
        let (outgoing, _rx) = mpsc::channel(2);
        let (_tx, incoming) = mpsc::channel(2);
        Arc::new(Slot {
            node_id: "node".into(),
            server_instance_id: format!("server-{instance}"),
            slot_id: "same-slot".into(),
            connection_id: connection.into(),
            closed: tokio::sync::watch::channel(None).0,
            outgoing,
            incoming: Mutex::new(incoming),
        })
    }

    #[tokio::test]
    async fn old_connection_cleanup_cannot_remove_replacement() {
        let pool = NodePool::new(config()).unwrap();
        pool.record_status(None, "old-control", status("one", true, "cpu"))
            .await
            .unwrap();
        let old = slot("one", "old");
        pool.register_slot(old).await.unwrap();
        let guard = pool.acquire(None, "old-request").await.unwrap();
        pool.remove_slot("node", "same-slot", "old", "closed").await;
        let new = slot("one", "new");
        pool.register_slot(new.clone()).await.unwrap();
        pool.record_status(None, "new-control", status("one", true, "cpu"))
            .await
            .unwrap();
        pool.remove_slot("node", "same-slot", "old", "late close")
            .await;
        pool.control_disconnected("node", "old-control", "late close")
            .await;
        assert!(pool
            .record_status(Some("node"), "old-control", status("one", true, "cpu"))
            .await
            .is_err());
        guard.finish().await;
        let state = pool.inner.state.lock().await;
        let node = &state.nodes[0];
        assert_eq!(node.control_connection_id.as_deref(), Some("new-control"));
        assert_eq!(node.slots["same-slot"].connection_id, "new");
        assert_eq!(node.idle_slots.len(), 1);
        assert_eq!(node.in_flight, 0);
        assert!(new.closed.borrow().is_none());
    }

    #[tokio::test]
    async fn instance_changes_invalidate_slots_without_corrupting_active_counts() {
        for supervisor_changed in [false, true] {
            let pool = NodePool::new(config()).unwrap();
            pool.record_status(None, "control", status("one", true, "cpu"))
                .await
                .unwrap();
            let old = slot("one", "old");
            pool.register_slot(old.clone()).await.unwrap();
            let guard = pool.acquire(None, "old-request").await.unwrap();
            let mut replacement = status("two", true, "cpu");
            if supervisor_changed {
                replacement.server_instance_id = "server-one".into();
            } else {
                replacement.supervisor_instance_id = "supervisor-one".into();
            }
            pool.record_status(Some("node"), "control", replacement)
                .await
                .unwrap();
            assert!(old.closed.borrow().is_some());
            let instance = if supervisor_changed { "one" } else { "two" };
            if !supervisor_changed {
                assert!(pool.register_slot(slot("one", "stale")).await.is_err());
            }
            pool.register_slot(slot(instance, "replacement"))
                .await
                .unwrap();
            let current = pool.acquire(None, "new-request").await.unwrap();
            guard.finish().await;
            let snapshot = pool.snapshot().await;
            assert_eq!(snapshot["active_requests"], 1);
            assert_eq!(snapshot["workers"][0]["available_capacity"], 0);
            current.finish().await;
            assert_eq!(pool.snapshot().await["workers"][0]["available_capacity"], 1);
        }
    }

    #[tokio::test]
    async fn retention_removes_data_only_nodes_rebuilds_indices_and_protects_requests() {
        let pool = NodePool::new(config()).unwrap();
        pool.register_slot(slot("one", "orphan")).await.unwrap();
        pool.record_status(None, "control", status("one", true, "cpu"))
            .await
            .unwrap();
        let guard = pool.acquire(None, "active").await.unwrap();
        pool.control_disconnected("node", "control", "gone").await;
        {
            let mut state = pool.inner.state.lock().await;
            ensure_node(&mut state, "data-only");
            ensure_node(&mut state, "keep");
            for node in &mut state.nodes[..2] {
                node.disconnected_since = Some(Instant::now() - Duration::from_secs(61));
            }
        }
        pool.expire_stale_nodes().await;
        {
            let state = pool.inner.state.lock().await;
            assert_eq!(state.nodes.len(), 2);
            assert_eq!(state.node_indices["keep"], 1);
            assert!(state.node_indices.contains_key("node"));
        }
        guard.finish().await;
        pool.expire_stale_nodes().await;
        {
            let mut state = pool.inner.state.lock().await;
            assert_eq!(state.node_indices["keep"], 0);
            state.nodes[0].disconnected_since = Some(Instant::now() - Duration::from_secs(61));
        }
        pool.expire_stale_nodes().await;
        assert!(pool.inner.state.lock().await.cohort.is_none());
        pool.record_status(None, "replacement", status("two", true, "different-arch"))
            .await
            .unwrap();
        pool.control_disconnected("node", "control", "old callback")
            .await;
        assert_eq!(pool.snapshot().await["workers"][0]["status"], "ready");
    }

    #[tokio::test]
    async fn queue_permits_return_on_cancellation_and_timeout() {
        let mut cfg = config();
        cfg.queue_wait_timeout = Duration::from_millis(20);
        let pool = NodePool::new(cfg).unwrap();
        let queued_pool = pool.clone();
        let task = tokio::spawn(async move { queued_pool.acquire(None, "cancelled").await });
        while pool.snapshot().await["queue_length"] != 1 {
            tokio::task::yield_now().await;
        }
        assert!(matches!(
            pool.acquire(None, "overflow").await,
            Err(AcquireError::QueueFull)
        ));
        task.abort();
        let _ = task.await;
        assert_eq!(pool.snapshot().await["queue_length"], 0);
        assert!(matches!(
            pool.acquire(None, "timeout").await,
            Err(AcquireError::TimedOut)
        ));
        assert_eq!(pool.snapshot().await["queue_length"], 0);
    }

    #[tokio::test]
    async fn cancellation_releases_capacity_even_with_a_full_outgoing_queue() {
        let pool = NodePool::new(config()).unwrap();
        pool.record_status(None, "control", status("one", true, "cpu"))
            .await
            .unwrap();
        let mut busy = slot("one", "busy");
        let (tx, _rx) = mpsc::channel(1);
        Arc::get_mut(&mut busy).unwrap().outgoing = tx;
        busy.outgoing.try_send(Ok(SlotFrame::default())).unwrap();
        pool.register_slot(busy.clone()).await.unwrap();
        let guard = pool.acquire(None, "cancel").await.unwrap();
        drop(guard);
        tokio::time::timeout(Duration::from_secs(1), async {
            while pool.snapshot().await["active_requests"] != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(busy.closed.borrow().is_some());
        assert_eq!(busy.close_status().message(), "client_disconnected");
    }
}
