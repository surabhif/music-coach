#!/usr/bin/env python3
"""Sync Gradio Space files to https://huggingface.co/spaces/surabhif/music-coach via huggingface_hub (Xet).

Git push rejects large binaries on HF; HfApi.upload_folder uses Xet storage and works for experts/*.wav.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

HF_SPACE = os.environ.get("HF_SPACE", "surabhif/music-coach")
ROOT = Path(__file__).resolve().parents[1]

# Minimal set for the Gradio / ZeroGPU Space (Flask/Docker stay on GitHub only).
INCLUDE_FILES = [
    "README.md",          # must keep YAML frontmatter (sdk: gradio, app_file, …)
    "gradio_app.py",
    "analysis.py",
    "paths.py",
    "requirements.txt",
    "packages.txt",
    ".gitignore",
]
INCLUDE_GLOBS = [
    "experts/*.wav",      # demo tracks (binary → Xet via huggingface_hub)
]

# Remove GitHub-/Flask-only paths if they ever landed on the Space.
DELETE_PATTERNS = [
    "screens/**",
    ".github/**",
    "templates/**",
    "static/**",
    "app.py",
    "db.py",
    "Dockerfile",
    "render.yaml",
    "shot.py",
    "shot_*.py",
    "test_*.py",
    "test_user_*.wav",
    "make_test_audio.py",
    "voice_synth.py",
    ".env.example",
    "scripts/**",
]


def _stage(staging: Path) -> None:
    for rel in INCLUDE_FILES:
        src = ROOT / rel
        if not src.is_file():
            raise FileNotFoundError(f"missing required Space file: {rel}")
        dest = staging / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    for pattern in INCLUDE_GLOBS:
        matched = list(ROOT.glob(pattern))
        if not matched:
            raise FileNotFoundError(f"no files matched {pattern}")
        for src in matched:
            rel = src.relative_to(ROOT)
            dest = staging / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    readme = (staging / "README.md").read_text(encoding="utf-8")
    if not readme.lstrip().startswith("---"):
        raise RuntimeError("README.md missing YAML frontmatter required by Gradio Spaces")
    yaml_block = readme.split("---", 2)[1]
    for key in ("sdk: gradio", "app_file: gradio_app.py"):
        if key not in yaml_block:
            raise RuntimeError(f"README YAML must include `{key}`")
    # HF validate-yaml only allows these thumbnail colors (not e.g. "rose").
    allowed = {"red", "yellow", "green", "blue", "indigo", "purple", "pink", "gray"}
    for field in ("colorFrom", "colorTo"):
        for line in yaml_block.splitlines():
            if line.strip().startswith(f"{field}:"):
                val = line.split(":", 1)[1].strip().strip("\"'")
                if val not in allowed:
                    raise RuntimeError(
                        f"README YAML {field}={val!r} invalid; must be one of {sorted(allowed)}"
                    )


def _report_runtime(api, repo_id: str) -> None:
    try:
        rt = api.get_space_runtime(repo_id)
    except Exception as exc:  # noqa: BLE001
        print(f"Space runtime unavailable yet: {exc}")
        return
    stage = getattr(rt, "stage", None) or getattr(rt, "status", None)
    hardware = getattr(rt, "hardware", None)
    print(f"Space runtime: stage={stage!r} hardware={hardware!r}")
    for attr in ("error_message", "dev_mode", "python_version"):
        if hasattr(rt, attr) and getattr(rt, attr) not in (None, ""):
            print(f"  {attr}={getattr(rt, attr)!r}")


def main() -> int:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) is required", file=sys.stderr)
        return 1

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    with tempfile.TemporaryDirectory(prefix="vocalmatch-space-") as tmp:
        staging = Path(tmp)
        _stage(staging)
        files = sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file())
        print(f"Uploading {len(files)} files to spaces/{HF_SPACE}:")
        for f in files:
            print(f"  - {f}")

        info = api.upload_folder(
            folder_path=str(staging),
            repo_id=HF_SPACE,
            repo_type="space",
            commit_message="Sync Gradio Space from GitHub (huggingface_hub / Xet)",
            delete_patterns=DELETE_PATTERNS,
            ignore_patterns=[".DS_Store"],
        )
        print(f"Upload commit: {info}")

    _report_runtime(api, HF_SPACE)
    try:
        meta = api.space_info(HF_SPACE)
        card = getattr(meta, "cardData", None) or getattr(meta, "card_data", None) or {}
        if hasattr(card, "get"):
            print(
                "Space card:",
                f"sdk={card.get('sdk')!r}",
                f"app_file={card.get('app_file')!r}",
                f"sdk_version={card.get('sdk_version')!r}",
            )
        elif card:
            print(f"Space cardData: {card}")
    except Exception as exc:  # noqa: BLE001
        print(f"space_info unavailable: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
