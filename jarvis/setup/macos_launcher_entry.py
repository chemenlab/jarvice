"""Native-bundle entry point for installer-managed macOS applications.

The compiled stub launcher (``macos_stub_launcher.c``) embeds the active
Python runtime in the Mach-O app process and then executes this source file
in place. Keeping the code outside the bundle lets normal source updates
apply without rebuilding or changing the TCC identity of the installed
application.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from importlib import import_module
from pathlib import Path


_LOCAL_LAUNCH_ENV = frozenset(
    {"JARVIS_CONFIG", "JARVIS_DATA_DIR", "LOCALAPPDATA", "JARVIS_INSTANCE", "JARVIS_BIND_HOST"}
)


def _apply_local_launch_profile(profile_path: Path) -> None:
    """Apply checkout-local defaults without reading config or credentials."""
    try:
        contents = profile_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        # Normal installs do not need a local launch profile.
        return
    except (OSError, UnicodeError) as exc:
        raise ValueError("could not read the profile as UTF-8") from exc

    try:
        profile = json.loads(contents)
    except json.JSONDecodeError as exc:
        raise ValueError("expected valid JSON") from exc
    if not isinstance(profile, dict):
        raise ValueError("expected an object of environment defaults")
    if profile.keys() - _LOCAL_LAUNCH_ENV:
        raise ValueError("contains unsupported environment keys")
    for value in profile.values():
        if not isinstance(value, str) or not value.strip() or "\0" in value:
            raise ValueError("values must be nonempty strings without NUL characters")

    # Validate the whole profile before applying any defaults. Explicit launch
    # environment (including an explicitly empty value) always takes priority.
    for key, value in profile.items():
        os.environ.setdefault(key, value)


def _write_identity_probe(destination: Path) -> int:
    """Record the native identity and managed-checkout import target."""
    try:
        from Foundation import NSBundle

        bundle = NSBundle.mainBundle()
        # Import the real launcher without starting it.  A py2app alias can
        # retain a valid Mach-O signature while its editable environment has
        # become unusable (for example after a Python/venv replacement).  The
        # installer probe must catch that before preserving the old bundle.
        launcher = import_module("jarvis.ui.web.launcher")
        install_root = Path(__file__).resolve().parents[2]
        payload = {
            "bundle_id": str(bundle.bundleIdentifier() or ""),
            "bundle_path": str(bundle.bundlePath() or ""),
            "executable": str(bundle.executablePath() or ""),
            "launcher_file": str(getattr(launcher, "__file__", "") or ""),
            "install_root": str(install_root),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "machine": platform.machine(),
        }
        destination.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - the installer treats this as failure
        payload = {"error": type(exc).__name__, "detail": str(exc)[:500]}
        destination.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    """Start the desktop launcher from the installer-managed checkout."""
    install_root = Path(__file__).resolve().parents[2]
    profile_path = install_root / ".jarvis-local-launch.json"
    try:
        _apply_local_launch_profile(profile_path)
    except ValueError as exc:
        print(f"Invalid local launch profile {profile_path}: {exc}", file=sys.stderr)
        return 2

    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        probe_index = arguments.index("--jarvis-identity-probe")
        probe_path = Path(arguments[probe_index + 1])
    except (ValueError, IndexError):
        probe_path = None
    if probe_path is not None:
        return _write_identity_probe(probe_path)

    os.chdir(install_root)

    # Resolve the editable managed checkout only at runtime. Keeping this
    # import dynamic prevents the alias builder from trying to package the
    # entire editable distribution into the identity-only launcher.
    launch_desktop = import_module("jarvis.ui.web.launcher").main
    return launch_desktop()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
