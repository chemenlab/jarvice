# Install this Russian assistant fork on another computer

Source: **https://github.com/chemenlab/jarvice**, branch **main**.
This fork adds Russian UI/speech, a shared Codex ChatGPT subscription brain,
live model selection, persistent history, and memory/procedure improvements.
Use the source installer below; upstream release downloads do not contain
these changes. The repository contains no transferable login or personal memory.

## macOS (Apple Silicon recommended) or Linux desktop

Install Git, Python 3.12, and Node.js 22+ using your usual package manager.
Python 3.12 is the version tested for this profile. Then run:

```bash
git clone --branch main https://github.com/chemenlab/jarvice.git
cd jarvice
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip rich packaging
python install/installer.py --no-launch
python scripts/configure_russian_assistant.py
```

The installer installs the full supported dependency profile and creates the
local desktop launcher. The profile script downloads multilingual Whisper
**base**, enables “Привет, Джарвис”, selects Russian and GPT-5.6-Sol, keeps
conversation history without age-based pruning, and disables login autostart.
It generates paths for this computer through the atomic configuration writer;
no path from the developer's machine is used. Run it before the first launch.
If the download fails, rerun it; configuration changes happen only after the
model is present. A prepared model directory can be supplied with `--wake-model`.

On macOS, launch the installed app from Spotlight, or:

```bash
open "$HOME/Applications/Personal Jarvis.app"
```

The source installer signs its launcher locally. Approve the native macOS
Microphone permission for voice, and Screen Recording/Accessibility when
enabling computer control. These grants cannot be copied from another Mac.
See [macOS packaging and permissions](../packaging/macos/README.md).

On Linux desktop, open Personal Jarvis from the application menu, or run:

```bash
.venv/bin/python -m jarvis.ui.web.launcher
```

## Windows x64

Install Git, Python 3.12, and Node.js 22+, then use PowerShell:

```powershell
git clone --branch main https://github.com/chemenlab/jarvice.git
cd jarvice
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip rich packaging
.venv\Scripts\python.exe install\installer.py --no-launch
.venv\Scripts\python.exe scripts\configure_russian_assistant.py
.venv\Scripts\python.exe -m jarvis.ui.web.launcher
```

Windows ARM64 currently lacks the required local Whisper/Vosk wheels. Use
manual voice activation or a supported x64 environment for this wake profile.
For other platform limitations see [OS parity](os-parity.md).

## Connect accounts on the new computer

1. Install Codex CLI if the installer has not provided it:
   `npm install -g @openai/codex`. Run `codex login` and choose your ChatGPT
   account. CLI 0.153.4 was verified for this integration. Do not copy
   `~/.codex/auth.json`, keychain records, tokens, or another machine's cache.
2. Open **API Keys → Main brain** and check that Codex reports a ChatGPT login.
   Select **GPT-5.6-Sol** in the model picker; available models depend on the
   account. Both GPT-5.5 and GPT-5.6-Sol passed tool and image round trips.
3. Enter the Gemini key in the app's Gemini provider card. It is used for
   speech recognition and speech synthesis; main reasoning stays on Codex.
   Secret storage uses the app's supported keyring/environment/file backend.
   Never put a key in a chat, source file, or `jarvis.toml`.
4. Say **“Привет, Джарвис”**, wait for activation, and ask a short question.
   Check microphone selection and permissions if activation fails. The
   multilingual Whisper path is used because the Russian Vosk vocabulary
   does not include this particular name.

Text chat remains usable when voice credentials or permissions are missing.
Background memory curation uses GPT-5.6-Luna; the main picker controls the
conversation model. The chosen account must have access to the selected model.

The generated profile includes these non-secret choices:

```toml
[ui]
language = "ru"
[brain]
primary = "codex-subscription"
reply_language = "ru"
[voice]
mode = "pipeline"
[trigger.wake_word]
phrase = "Привет, Джарвис"
engine = "stt_match"
language = "ru"
[sessions]
retention_days = 0
```

## Updates and verification limits

Update this checkout with ordinary Git after saving local work. Do not use an
upstream release installer to update this fork. No release binaries are
published as part of this source snapshot.

Verified on macOS arm64: native startup, Russian chat, Codex subscription
tools and user/tool images, memory recall and dated correction across chats
and a new process, recorded Russian wake audio, frontend build, and affected
tests. See [protocol evidence](codex-subscription-protocol.md) and
[OS contract evidence](os-parity.md). Windows/Linux transport behavior is
emulated in contract tests; a physical installation on those systems was not
performed. Physical microphone wake/conversation and a complete desktop
clarification/procedure replay remain acceptance checks on the target machine.
