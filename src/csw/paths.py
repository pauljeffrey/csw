"""Where data and secrets live.

Nothing in this repository stores data.  Every stage reads and writes below one *data root* and takes secrets from the
process environment.

CSW_DATA_ROOT   directory that holds seeds/, anchors/, exemplars/, work/, out*/ ...   (default: the current directory)
CSW_ENV_FILE    optional KEY=VALUE file that is read into os.environ (existing variables win)   (default: <data root>/.env)
CSW_PAPER_DIR   directory with the judge prompt card and its JSON schema               (default: <data root>/paper)
"""
from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("CSW_DATA_ROOT", ".")).resolve()
PAPER_DIR = Path(os.environ.get("CSW_PAPER_DIR", str(DATA_ROOT / "paper")))
ENV_FILE = Path(os.environ.get("CSW_ENV_FILE", str(DATA_ROOT / ".env")))


def load_env_file(path: Path | str | None = None) -> None:
    """Read KEY=VALUE lines from `path` into os.environ without overriding variables that are already set."""
    p = Path(path) if path else ENV_FILE
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def env_value(name: str, default: str | None = None) -> str | None:
    """Value of environment variable `name` (the optional env file is read first)."""
    load_env_file()
    return os.environ.get(name, default)


SRC_DIR = Path(__file__).resolve().parents[1]


def subprocess_env(**extra: str) -> dict:
    """Environment for child processes that run `python -m csw.<stage>.<module>`: the current environment with this
    source tree on PYTHONPATH (so it works with or without an editable install)."""
    env = dict(os.environ, **extra)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(SRC_DIR), env.get("PYTHONPATH", "")) if p)
    return env
