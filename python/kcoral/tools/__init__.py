"""Built-in remote tools, with one definition module per command."""

# CLI spelling -> module name. Imports stay lazy so help needs only the client.
COMMANDS = {
    "python": "python",
    "compute-sanitizer": "compute_sanitizer",
    "ncu": "ncu",
    "run-iket": "run_iket",
    "bench": "bench",
    "shell": "shell",
}
