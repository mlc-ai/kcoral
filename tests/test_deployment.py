from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_compose_publishes_server_only_on_loopback():
    compose = (ROOT / "docker" / "compose.yaml").read_text()
    assert '"127.0.0.1:${BENCH_PORT:-8000}:8000"' in compose
    assert '- "${BENCH_PORT:-8000}:8000"' not in compose


def test_image_uses_cli_loopback_default():
    dockerfile = (ROOT / "docker" / "Dockerfile.cu130.dev").read_text()
    assert "BENCH_HOST=0.0.0.0" not in dockerfile
