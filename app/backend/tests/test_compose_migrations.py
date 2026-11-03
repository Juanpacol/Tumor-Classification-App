import pathlib
import re

import pytest
import yaml

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
COMPOSE_FILE = BACKEND_ROOT.parent.parent / "docker-compose.yml"


def _backend_command(compose_file: pathlib.Path) -> str:
    command = yaml.safe_load(compose_file.read_text())["services"]["backend"]["command"]
    return command if isinstance(command, str) else " ".join(command)


@pytest.mark.skipif(
    not COMPOSE_FILE.exists(), reason="repo-root docker-compose.yml not mounted"
)
def test_compose_backend_migrates_before_starting_uvicorn() -> None:
    command = _backend_command(COMPOSE_FILE)
    assert re.search(r"alembic upgrade head\s*&&\s*exec uvicorn\b", command), command
