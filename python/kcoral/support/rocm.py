"""AMD ROCm device identity, visibility and HIP runtime error APIs.

Devices are read from the KFD topology in sysfs rather than a management
library: it is always present where ``/dev/kfd`` is, needs no extra package,
and is what the ROCm runtime (ROCr) itself enumerates. A GPU id is an index into
ROCr's order: KFD nodes by number, CPU nodes skipped, and nodes whose render
device this process cannot open skipped too (a container may expose a subset).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

TOPOLOGY = Path("/sys/class/kfd/kfd/topology/nodes")

# ``/dev/kfd`` serves every GPU, so a worker must not run unless it verifiably
# came up on the card it was given.
STRICT_IDENTITY = True
# ROCr filters devices below HIP, so HIP's own variables would then index the
# filtered set. Clear them rather than let a value the server inherited apply.
VISIBILITY_VARIABLES = ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
RUNTIME_PATHS = ("/opt/rocm",)


@dataclass(frozen=True)
class Agent:
    """One GPU node of the KFD topology."""

    node: int
    unique_id: int
    domain: int
    location_id: int  # PCI bus << 8 | device << 3 | function
    render_minor: int

    @property
    def pci(self) -> str:
        bus, device = (self.location_id >> 8) & 0xFF, (self.location_id >> 3) & 0x1F
        return f"{self.domain:04x}:{bus:02x}:{device:02x}"

    @property
    def render_node(self) -> Path:
        return Path(f"/dev/dri/renderD{self.render_minor}")


def _properties(node: Path) -> dict[str, int]:
    properties = {}
    for line in (node / "properties").read_text().splitlines():
        key, _, value = line.partition(" ")
        try:
            properties[key] = int(value)
        except ValueError:
            continue
    return properties


@cache
def agents(topology: Path = TOPOLOGY) -> tuple[Agent, ...]:
    """The GPUs this process can use, in ROCr order; empty where KFD is absent."""
    try:
        nodes = sorted(
            (p for p in topology.iterdir() if p.name.isdigit()), key=lambda p: int(p.name)
        )
    except OSError:
        return ()
    found = []
    for node in nodes:
        try:
            if int((node / "gpu_id").read_text().strip() or 0) == 0:
                continue  # a CPU node
            properties = _properties(node)
        except (OSError, ValueError):
            continue
        agent = Agent(
            node=int(node.name),
            unique_id=properties.get("unique_id", 0),
            domain=properties.get("domain", 0),
            location_id=properties.get("location_id", 0),
            render_minor=properties.get("drm_render_minor", -1),
        )
        if agent.render_minor < 0 or not os.access(agent.render_node, os.R_OK | os.W_OK):
            continue  # ROCr skips a GPU it cannot open
        found.append(agent)
    return tuple(found)


def _agent(gpu_id: int) -> Agent | None:
    found = agents()
    return found[gpu_id] if 0 <= gpu_id < len(found) else None


def device_selector(gpu_id: int) -> str:
    """How ``ROCR_VISIBLE_DEVICES`` names a physical GPU.

    The UUID is spelled as ``rocminfo`` prints it. Unlike an index, it still
    names the same card inside a sandbox that exposes only some render nodes.
    """
    agent = _agent(gpu_id)
    if agent is None or agent.unique_id == 0:
        return str(gpu_id)
    return f"GPU-{agent.unique_id:016x}"


def device_identity(gpu_id: int) -> str | None:
    """The PCI address a worker for this GPU must report.

    HIP's device UUID is not spelled consistently across ROCm and torch
    releases; the PCI location is.
    """
    agent = _agent(gpu_id)
    return None if agent is None else agent.pci


def identity_key(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().lower() or None


def visible_devices_env(selector: str) -> dict[str, str | None]:
    return {
        "ROCR_VISIBLE_DEVICES": selector,
        "HIP_VISIBLE_DEVICES": None,
        "CUDA_VISIBLE_DEVICES": None,
        "GPU_DEVICE_ORDINAL": None,
    }


def device_nodes(gpu_ids: tuple[int, ...]) -> list[Path]:
    """``/dev/kfd`` plus only the assigned render nodes.

    ``/dev/kfd`` itself is shared by every GPU; without a card's render node,
    ROCr cannot create a queue on it and leaves it out.
    """
    devices = [Path("/dev/kfd")]
    for gpu in gpu_ids:
        agent = _agent(gpu)
        if agent is not None:
            devices.append(agent.render_node)
    return devices


def describe_device_identity() -> str | None:
    """The visible GPU's PCI address, inside a worker; None if torch lacks it."""
    import torch

    try:
        properties = torch.cuda.get_device_properties(0)
        return (
            f"{int(properties.pci_domain_id):04x}:{int(properties.pci_bus_id):02x}:"
            f"{int(properties.pci_device_id):02x}"
        )
    except Exception:  # no such attributes on older torch, or no visible device
        return None


class _HIPErrorAPI:
    """The small libamdhip64 surface needed to inspect HIP's last-error slot."""

    def __init__(self, library: Any) -> None:
        self._get_last_error = library.hipGetLastError
        self._get_last_error.argtypes = []
        self._get_last_error.restype = ctypes.c_int
        self._get_error_name = library.hipGetErrorName
        self._get_error_name.argtypes = [ctypes.c_int]
        self._get_error_name.restype = ctypes.c_char_p
        self._get_error_string = library.hipGetErrorString
        self._get_error_string.argtypes = [ctypes.c_int]
        self._get_error_string.restype = ctypes.c_char_p

    def take_last_error(self) -> str | None:
        code = self._get_last_error()
        if code == 0:
            return None
        name = _decode(self._get_error_name(code), "hipErrorUnknown")
        description = _decode(self._get_error_string(code), "unknown error")
        return f"HIP error {name} ({code}): {description}"


def _decode(value: bytes | None, fallback: str) -> str:
    return value.decode("utf-8", errors="replace") if value is not None else fallback


@cache
def _hip_error_api() -> _HIPErrorAPI:
    """Load the HIP runtime torch uses, once per worker process.

    The last-error slot is per runtime instance, so prefer the copy bundled in
    the torch wheel over a system one.
    """
    candidates: list[str] = []
    try:
        import torch

        candidates += [
            str(p) for p in sorted((Path(torch.__file__).parent / "lib").glob("libamdhip64.so*"))
        ]
    except Exception:
        pass
    discovered = ctypes.util.find_library("amdhip64")
    if discovered is not None:
        candidates.append(discovered)
    candidates += [
        "libamdhip64.so",
        "libamdhip64.so.7",
        "libamdhip64.so.6",
        "/opt/rocm/lib/libamdhip64.so",
    ]

    failures: list[str] = []
    for candidate in dict.fromkeys(candidates):
        try:
            return _HIPErrorAPI(ctypes.CDLL(candidate))
        except (AttributeError, OSError) as exc:
            failures.append(f"{candidate}: {exc}")
    raise RuntimeError("could not load libamdhip64 to inspect HIP errors: " + "; ".join(failures))


def take_last_error() -> str | None:
    return _hip_error_api().take_last_error()
