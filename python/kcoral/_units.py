"""Convert user-facing binary megabytes to internal byte budgets."""


def mbytes_to_bytes(value: float, name: str, *, positive: bool = False) -> int:
    """Convert MiB to whole bytes, rejecting invalid or unrepresentable sizes."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 2**44:
        raise ValueError(f"{name} must be a finite non-negative number below 2**44 MiB")
    size = int(value * 1024**2)
    if (value > 0 or positive) and size == 0:
        raise ValueError(f"{name} must be at least one byte (1 / 1048576 MiB)")
    return size
