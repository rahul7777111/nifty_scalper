# Workspace Init Checklist

- [x] Verify [.github/copilot-instructions.md](.github/copilot-instructions.md) exists
- [x] Clarify project requirements (existing Python project)
- [x] Scaffold project (already present)
- [x] Customize project (not requested in /init)
- [x] Install required extensions (none specified)
- [ ] Compile project and run diagnostics (blocked: Python runtime unavailable in shell)
- [x] Create and run task (skipped for /init)
- [ ] Launch project (pending explicit debug-mode confirmation)
- [x] Ensure documentation exists and remove HTML comments from this file

Notes:
- [README.md](README.md) is present.
- `python -m pytest -q` failed because `python` is not available on PATH.
- `.\.venv\Scripts\python.exe -m pytest -q` failed because the venv interpreter target path is invalid.
