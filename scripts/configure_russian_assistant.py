"""Prepare the Russian subscription profile before the first desktop launch.

Run from the installed checkout's virtual environment. All configuration writes
go through config_writer; credentials stay in the app's supported secret store.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def configure(path: Path, wake_model: Path, *, model: str = "gpt-5.6-sol") -> None:
    """Apply the profile without launching, stopping, or restarting any app."""
    from jarvis.core import config_writer as writer

    writer.set_subscription_main_brain(model=model, path=path)
    writer.set_ui_language("ru", path=path)
    writer.set_reply_language("ru", path=path)
    writer.set_stt_language("ru", path=path)
    writer.set_wake_word("Привет, Джарвис", engine="stt_match", path=path)
    writer.set_wake_language("ru", path=path)
    writer.set_wake_word_enabled(True, path=path)
    writer.set_autostart(False, path=path)
    writer._patch_table(path, "sessions", "retention_days", 0)
    writer._patch_table(path, "stt", "wake_model", str(wake_model.resolve()), extra={
        "wake_device": "cpu", "wake_compute_type": "int8",
    })
    writer._patch_wiki_curator_toml(path, {
        "provider": "codex-subscription", "model": "gpt-5.6-luna",
    })


def main() -> None:
    from jarvis.core.config import resolve_config_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=resolve_config_path())
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument(
        "--wake-model", type=Path,
        help="Use an already downloaded multilingual faster-whisper model directory.",
    )
    args = parser.parse_args()
    model_path = args.wake_model
    if model_path is None:
        try:
            from faster_whisper.utils import download_model
        except ImportError as exc:
            raise SystemExit(
                'Local wake dependencies are missing. Install with: pip install -e ".[local-voice]"'
            ) from exc
        destination = args.config.resolve().parent / "data" / "wake_models" / "whisper" / "base"
        print("Downloading the multilingual Whisper base wake model (first run only).")
        model_path = Path(download_model("base", output_dir=str(destination)))
    if not (model_path / "model.bin").is_file() or not (model_path / "config.json").is_file():
        raise SystemExit("Wake model directory is incomplete; configuration was not changed.")
    configure(args.config, model_path, model=args.model)
    print("Russian profile saved. Start the app and sign in to Codex.")
    print("Add the Gemini speech key in the app's API Keys view.")
    print("Login autostart is disabled. Existing application processes were not changed.")


if __name__ == "__main__":
    main()
