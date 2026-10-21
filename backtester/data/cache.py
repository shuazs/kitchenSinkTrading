import os
from pathlib import Path


def cache_dir(sub: str = "") -> Path:
    root = Path(os.environ.get("KS_CACHE_DIR", Path.home() / ".cache" / "kitchensink"))
    path = root / sub if sub else root
    path.mkdir(parents=True, exist_ok=True)
    return path
