from abc import ABC, abstractmethod
from pathlib import Path
import re
import subprocess


def resolve_source(path):
    """Resolve a local file or a Git LFS pointer in its repository's default cache."""
    path = Path(path)
    with path.open("rb") as source:
        if source.readline(64).strip() != b"version https://git-lfs.github.com/spec/v1":
            return path
        oid = re.fullmatch(rb"oid sha256:([0-9a-f]{64})", source.readline(80).strip())
        size = re.fullmatch(rb"size ([0-9]+)", source.readline(64).strip())
    if oid is None or size is None:
        raise ValueError(f"Invalid Git LFS pointer in {path}.")

    try:
        git_dir = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=path.absolute().parent, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise FileNotFoundError(f"Cannot locate the Git repository for {path}.") from error

    digest = oid[1].decode("ascii")
    cached = Path(git_dir) / "lfs" / "objects" / digest[:2] / digest[2:4] / digest
    if not cached.is_file() or cached.stat().st_size != int(size[1]):
        raise FileNotFoundError(f"Restore the Git LFS content for {path} before running.")
    return cached


class DataSource(ABC):

    @abstractmethod
    def xy(self):
        pass

    @abstractmethod
    def train_test(self):
        pass

    @abstractmethod
    def get_cv_split_method(self):
        pass
