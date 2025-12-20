# Repository Guidelines

## Project Structure & Module Organization
- `frameloop/cli.py`: Typer-based CLI entrypoint (`frameloop` script) exposing `video`, `image`, `upscale`, `models`, `info`, and `status` commands.
- `frameloop/models.py`: Model registry and metadata (IDs, params, pricing notes).
- `frameloop/runner.py`: Prediction execution loop, polling, and output handling.
- `frameloop/utils.py`: Helpers for URL detection, file handling, downloads, and duration formatting.
- `pyproject.toml`, `uv.lock`: Project metadata, dependencies, and script entry definition.

## Build, Test, and Development Commands
- Install deps (Python 3.11+): `uv sync` (or `uv pip install -e .` if you prefer editable mode).
- Run the CLI: `uv run frameloop --help`, `uv run frameloop models`, or `uv run frameloop video <image> -p "<prompt>" --no-wait`.
- Lint: `uv run ruff check frameloop` (dev dependency defined in `pyproject.toml`).
- Package build (optional): `uv build`.

## Coding Style & Naming Conventions
- Python 3.11+, 4-space indentation, type hints preferred (existing functions are annotated).
- Keep CLI UX consistent with Typer’s patterns: descriptive help text, typed options, graceful error exits.
- Use descriptive, lowercase model keys in `MODELS`; keep parameter schemas concise and validated before use.
- Favor small, single-purpose helpers in `utils.py`; prefer rich console output for user-facing messaging.

## Testing Guidelines
- No automated test suite is present yet. Before merging, sanity-check key flows:
  - `frameloop video` with a local image path and `--no-wait`.
  - `frameloop image` with and without `--image` inputs.
  - `frameloop upscale` on a small PNG/JPG.
- When adding tests, colocate them under `tests/` and run with `uv run pytest` (add pytest to dev deps).

## Commit & Pull Request Guidelines
- Use concise, imperative commit subjects (e.g., `Add WAN fast-mode validation`, `Improve output filename generator`).
- For pull requests, include:
  - What changed and why, with links to issues or tasks.
  - CLI examples or screenshots of console output when UX changes.
  - Notes on manual tests run (commands, inputs, observed results) and any remaining risks.

## Security & Configuration Notes
- Set `REPLICATE_API_TOKEN` in your environment before running commands; avoid committing tokens or `.env` files.
- Validate file inputs early and fail with clear messages to prevent unintended network requests or missing files.
