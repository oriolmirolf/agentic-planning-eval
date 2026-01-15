from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class GitInfo:
    repo_root: str
    commit: str
    dirty: bool


def get_git_info(start_path: str | None = None) -> GitInfo:
    """
    Resolve git commit + dirty status. This is required for reproducibility.

    Uses gitpython's Repo(search_parent_directories=True).
    """
    from git import Repo  # type: ignore

    repo = Repo(path=start_path or os.getcwd(), search_parent_directories=True)
    commit = repo.head.commit.hexsha
    dirty = repo.is_dirty(untracked_files=True)
    repo_root = repo.working_tree_dir or os.getcwd()
    return GitInfo(repo_root=repo_root, commit=commit, dirty=dirty)


def init_mlflow(experiment_name: str, tracking_uri: str | None = None) -> None:
    import mlflow  # type: ignore

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def set_required_git_tag(git_commit: str) -> None:
    import mlflow  # type: ignore

    # Hard constraint: git commit hash must be logged as a tag named
    # 'git_commit'
    mlflow.set_tag("git_commit", git_commit)
