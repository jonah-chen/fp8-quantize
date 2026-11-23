"""Resolve the source model folder: a local path or a Hugging Face download."""
from __future__ import annotations

import sys
from pathlib import Path

from .utils import fail


def _looks_like_local_path(model: str) -> bool:
    """Heuristic: does the user mean a path on their machine?"""
    if model.startswith(("~", "/", "./", "../")) or "\\" in model:
        return True
    # "models/foo" without a leading ./ : treat as local when ./models exists
    path = Path(model)
    return len(path.parts) > 1 and path.parent.exists()


def confirm(prompt: str, *, assume_yes: bool) -> bool:
    """Ask the user to approve an action; return True if approved.

    With ``assume_yes`` the answer is a pre-emptive yes and nothing is asked.
    In a non-interactive environment (stdin is not a TTY) an unconfirmed
    action is an error, so batch runs must pass ``--yes`` explicitly.
    """
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        fail("this step needs an interactive prompt - run in a terminal, or add --yes")
    answer = input(f"{prompt} [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def resolve_source(
    model: str,
    *,
    assume_yes: bool,
    token: str | None = None,
    revision: str | None = None,
) -> Path:
    """Return a local folder containing the model, downloading from HF if needed.

    ``model`` is either an existing local model folder (used as-is, no
    download) or a Hugging Face repo id. A download is only attempted after
    interactive confirmation, unless ``assume_yes`` is set.
    """
    candidate = Path(model).expanduser()
    if candidate.is_dir():
        src = candidate.resolve()
        print(f"Using local model folder: {src} (no download needed)")
        return src
    if _looks_like_local_path(model):
        fail(f"'{model}' looks like a local path, but it does not exist")

    repo_id = model
    if not confirm(f"Download '{repo_id}' from the Hugging Face Hub?", assume_yes=assume_yes):
        sys.exit(
            "Aborted: the model was not downloaded. "
            "Re-run when ready (add --yes to skip the prompt)."
        )

    from huggingface_hub import snapshot_download

    print(f"Downloading {repo_id} from the Hugging Face Hub ...")
    local_dir = snapshot_download(repo_id, token=token, revision=revision)
    return Path(local_dir).resolve()
