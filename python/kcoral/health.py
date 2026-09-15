"""Public health response schema."""

from typing import Literal

from pydantic import BaseModel, Field


class RequestLoad(BaseModel):
    request_capacity: int = Field(
        ge=0,
        description="Serviceable request capacity, occupied and free.",
    )
    requests_in_progress: int = Field(
        ge=0,
        description="Assigned requests, including compilation, GPU waiting, and cleanup.",
    )
    requests_waiting: int = Field(
        ge=0,
        description="Requests awaiting assignment.",
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"] = Field(description="Endpoint health status.")
    instance_id: str = Field(description="Changes on each endpoint restart.")
    started_at: str = Field(description="Endpoint startup time in UTC (RFC 3339).")
    gpu_count: int | None = Field(description="Number of configured GPUs, when known.")
    load: RequestLoad
    target: dict[str, str] = Field(description="Compilation target.")
    versions: dict[str, str] = Field(description="Installed runtime and toolchain versions.")
