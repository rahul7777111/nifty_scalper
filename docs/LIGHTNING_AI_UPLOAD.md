# Lightning AI Upload

This repo is too large and too stateful to upload as-is from the project root.
Local folders such as `.venv/`, `data/`, `models/`, `reports/`, `logs/`, and secret files like `.env` or `.scalper.env` should stay out of the Studio upload.

## Recommended path

If your latest code is already on GitHub, use Lightning AI's GitHub import flow inside a Studio and clone:

```bash
git clone https://github.com/rahul7777111/NiftyScalper.git
```

That is the safest option because the current working tree contains many local-only files and generated artifacts.

## Local upload path

1. Install the Lightning CLI:

```bash
python -m pip install -U lightning-sdk
```

2. Authenticate:

```bash
lightning login
```

3. Build a clean upload bundle from this repo:

```powershell
powershell -ExecutionPolicy Bypass -File tools/prepare_lightning_upload.ps1
```

4. Upload the prepared folder to your Studio:

```bash
lightning upload dist/lightning_upload_bundle --studio <teamspace/studio_name> --recursive
```

## What the prep script includes

- `src/`
- `scripts/`
- `tests/`
- `tools/`
- `docs/`
- `config/`
- `pytradingapi-typeB-main/`
- `DhanHQ-py-main/`
- core project files like `README.md`, `requirements.txt`, `pyproject.toml`, `pytest.ini`, `conftest.py`

## What it excludes

- virtual environments and caches
- secrets and env files
- local databases and logs
- generated reports and model artifacts
- large local data folders
- temporary root-level scratch files such as `_*.py` and `tmp_*`

## After upload

Inside the Studio terminal, install dependencies and launch what you need:

```bash
pip install -r requirements.txt
python -m src.ui
```

If you only want headless research or tests, run the usual `scripts/` and `pytest` commands instead.
