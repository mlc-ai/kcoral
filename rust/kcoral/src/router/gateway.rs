use super::{pool::Slot, NodePool, FIRST_FRAME_TIMEOUT, GRPC_MESSAGE_BYTES, PROTOCOL_VERSION};
use crate::proto::{
    router_gateway_server::{RouterGateway, RouterGatewayServer},
    slot_frame::Payload,
    SlotFrame, SupervisorAck, SupervisorStatus,
};
use async_stream::try_stream;
use futures_util::Stream;
use std::{pin::Pin, sync::Arc};
use subtle::ConstantTimeEq;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response as GrpcResponse, Status, Streaming};
use uuid::Uuid;
type ControlStream = Pin<Box<dyn Stream<Item = Result<SupervisorAck, Status>> + Send + 'static>>;
type DataStream = Pin<Box<dyn Stream<Item = Result<SlotFrame, Status>> + Send + 'static>>;

#[derive(Clone)]
struct RouterGatewayService {
    pool: NodePool,
}

impl RouterGatewayService {
    fn authorized<T>(&self, request: &Request<T>) -> bool {
        let Some(expected) = self.pool.node_token() else {
            return true;
        };
        let supplied = request
            .metadata()
            .get("authorization")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("Bearer "));
        supplied.is_some_and(|supplied| bool::from(supplied.as_bytes().ct_eq(expected.as_bytes())))
    }
}

#[tonic::async_trait]
impl RouterGateway for RouterGatewayService {
    type ConnectSupervisorStream = ControlStream;
    type ConnectSlotStream = DataStream;

    async fn connect_supervisor(
        &self,
        request: Request<Streaming<SupervisorStatus>>,
    ) -> Result<GrpcResponse<Self::ConnectSupervisorStream>, Status> {
        if !self.authorized(&request) {
            return Err(Status::unauthenticated("invalid node bearer token"));
        }
        let mut stream = request.into_inner();
        let first = tokio::time::timeout(FIRST_FRAME_TIMEOUT, stream.message())
            .await
            .map_err(|_| Status::deadline_exceeded("timed out waiting for initial status"))??
            .ok_or_else(|| {
                Status::invalid_argument("control stream ended before initial status")
            })?;
        let connection_id = Uuid::new_v4().to_string();
        let node_id = self.pool.record_status(None, &connection_id, first).await?;
        let (commands_tx, commands_rx) = mpsc::channel(2);
        commands_tx
            .send(Ok(SupervisorAck {}))
            .await
            .map_err(|_| Status::unavailable("control response stream closed"))?;

        let pool = self.pool.clone();
        tokio::spawn(async move {
            let _keep_response_open = commands_tx;
            let reason = loop {
                match stream.message().await {
                    Ok(Some(status)) => {
                        if let Err(error) = pool
                            .record_status(Some(&node_id), &connection_id, status)
                            .await
                        {
                            break error.message().to_string();
                        }
                    }
                    Ok(None) => break "supervisor control stream ended".to_string(),
                    Err(error) => break format!("supervisor control stream failed: {error}"),
                }
            };
            pool.control_disconnected(&node_id, &connection_id, &reason)
                .await;
        });

        Ok(GrpcResponse::new(Box::pin(ReceiverStream::new(
            commands_rx,
        ))))
    }

    async fn connect_slot(
        &self,
        request: Request<Streaming<SlotFrame>>,
    ) -> Result<GrpcResponse<Self::ConnectSlotStream>, Status> {
        if !self.authorized(&request) {
            return Err(Status::unauthenticated("invalid node bearer token"));
        }
        let mut stream = request.into_inner();
        let first = tokio::time::timeout(FIRST_FRAME_TIMEOUT, stream.message())
            .await
            .map_err(|_| Status::deadline_exceeded("timed out waiting for data hello"))??
            .ok_or_else(|| Status::invalid_argument("data stream ended before hello"))?;
        if !first.request_id.is_empty() {
            return Err(Status::invalid_argument(
                "data hello must not carry a request identifier",
            ));
        }
        let Some(Payload::Hello(hello)) = first.payload else {
            return Err(Status::invalid_argument("first data frame must be hello"));
        };
        if hello.protocol_version != PROTOCOL_VERSION {
            return Err(Status::failed_precondition(
                "unsupported tunnel protocol version",
            ));
        }

        let connection_id = Uuid::new_v4().to_string();
        let (outgoing_tx, mut outgoing_rx) = mpsc::channel(2);
        let (incoming_tx, incoming_rx) = mpsc::channel(2);
        let slot = Arc::new(Slot::new(
            hello.node_id.clone(),
            hello.server_instance_id,
            hello.slot_id.clone(),
            connection_id.clone(),
            outgoing_tx,
            incoming_rx,
        ));
        // Queue acceptance before exposing the slot to HTTP dispatch.
        slot.send(SlotFrame {
            request_id: String::new(),
            payload: Some(Payload::Ack(crate::proto::SlotAck {})),
        })
        .await?;
        self.pool.register_slot(slot.clone()).await?;
        let mut closed = slot.subscribe_closed();
        let closing_slot = slot.clone();

        let pool = self.pool.clone();
        tokio::spawn(async move {
            let reason = loop {
                match stream.message().await {
                    Ok(Some(frame)) => {
                        if matches!(frame.payload, Some(Payload::Hello(_))) {
                            break "data stream sent a second hello".to_string();
                        }
                        if incoming_tx.send(frame).await.is_err() {
                            break "router stopped consuming the data slot".to_string();
                        }
                    }
                    Ok(None) => break "outbound data stream ended".to_string(),
                    Err(error) => break format!("outbound data stream failed: {error}"),
                }
            };
            pool.remove_slot(&hello.node_id, &hello.slot_id, &connection_id, &reason)
                .await;
        });

        let outgoing = try_stream! {
            loop {
                if closed.borrow().is_some() { Err(closing_slot.close_status())?; }
                let item = tokio::select! {
                    item = outgoing_rx.recv() => item,
                    _ = closed.changed() => Some(Err(closing_slot.close_status())),
                };
                match item { Some(frame) => yield frame?, None => break }
            }
        };
        Ok(GrpcResponse::new(Box::pin(outgoing)))
    }
}

pub(super) fn routes(pool: NodePool) -> axum::Router {
    let grpc = RouterGatewayServer::new(RouterGatewayService { pool })
        .max_decoding_message_size(GRPC_MESSAGE_BYTES)
        .max_encoding_message_size(GRPC_MESSAGE_BYTES);
    tonic::service::Routes::new(grpc).into_axum_router()
}
