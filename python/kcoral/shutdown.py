"""Coordinate Uvicorn shutdown with worker cleanup."""

from __future__ import annotations

import asyncio
import logging

import uvicorn

logger = logging.getLogger("uvicorn.error")


class ShutdownServer(uvicorn.Server):
    def __init__(self, config: uvicorn.Config, app) -> None:
        super().__init__(config)
        self.app = app

    def handle_exit(self, sig, frame) -> None:
        # force_exit skips lifespan; replaying signals can interrupt cleanup.
        self.should_exit = True
        asyncio.get_running_loop().call_soon(self._begin_shutdown)

    def _begin_shutdown(self) -> None:
        pool = getattr(self.app.state, "pool", None)
        if pool is not None:
            pool.begin_shutdown()
            logger.info(
                "Shutdown requested. Finishing remaining benchmarks: %d left.", pool.active_requests
            )

    async def shutdown(self, sockets=None) -> None:
        pool = getattr(self.app.state, "pool", None)
        if pool is not None and not pool.closing:
            self._begin_shutdown()
        progress = asyncio.create_task(self._report_progress(pool))
        try:
            await super().shutdown(sockets)
        finally:
            progress.cancel()
            await asyncio.gather(progress, return_exceptions=True)

    async def _report_progress(self, pool) -> None:
        remaining = pool.active_requests if pool is not None else 0
        while True:
            await asyncio.sleep(0.1)
            active = pool.active_requests if pool is not None else 0
            if active != remaining:
                logger.info("Finishing remaining benchmarks: %d left.", active)
                remaining = active

    async def _wait_tasks_to_complete(self) -> None:
        # Uvicorn's default wait advertises Ctrl+C to force quit, which skips cleanup.
        while self.server_state.connections or self.server_state.tasks:
            await asyncio.sleep(0.1)
        for server in self.servers:
            await server.wait_closed()
