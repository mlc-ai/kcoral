//! Bounded execution-slot transport and request ownership.
use super::{NodePool, DATA_CHUNK_BYTES};
use crate::proto::{slot_frame::Payload, SlotFrame};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tonic::Status;

pub(super) struct Slot {
    pub(super) node_id: String,
    pub(super) server_instance_id: String,
    pub(super) slot_id: String,
    // Identifies this connection, so a late old cleanup cannot remove a replacement.
    pub(super) connection_id: String,
    pub(super) closed: tokio::sync::watch::Sender<Option<String>>,
    pub(super) outgoing: mpsc::Sender<Result<SlotFrame, Status>>,
    pub(super) incoming: Mutex<mpsc::Receiver<SlotFrame>>,
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

    pub(super) fn disconnect(&self, reason: &str) {
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

pub(super) struct SlotGuard {
    pub(super) pool: NodePool,
    pub(super) node_id: String,
    pub(super) request_id: String,
    pub(super) slot: Arc<Slot>,
    pub(super) active: bool,
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
