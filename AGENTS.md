# Repository Guidelines

## Project Structure & Module Organization
- Entrypoints: `run.py` (supervised), `run_pretrain.py`, `run_fewshot.py`, `run_DG.py`, `run_tokenizer.py`.
- Experiment wiring: `exp/` (`exp_sup.py`, `exp_pretrain.py`, `exp_dg.py`, etc.) connects data, models, training, and eval.
- Models: `models/` (e.g., `RmGPT.py`, `ResNet1D.py`, `Transformer.py`) with shared blocks under `models/layers/`.
- Data: `data_provider/` (loaders/factories) and YAML configs in `data_provider/data_config/**` (e.g., `baseline/ROT.yaml`, `main_result/multi_task.yaml`).
- Utilities: `utils/` for augmentation, losses, metrics, DDP helpers.
- Outputs: checkpoints to `checkpoints/`, logs to `logs/` (gitignored), visualizations to `visualization_results/`.

## Build, Test, and Development Commands
- Create env: `conda env create -f environment.yml` then `conda activate time_series` (Python 3.9).
- Supervised train: `python run.py --is_training 1 --task_data_config_path data_provider/data_config/baseline/ROT.yaml`.
- Eval: `python run.py --is_training 0 --pretrained_weight checkpoints/<exp>/checkpoint.pth`.
- Pretrain sweep example: see `scripts/scale_exp.sh` (uses tmux; run under bash/WSL/Linux).
- Visualization: append `--visualize --extract_attention` to training/testing, or run `python visualize_analysis.py --checkpoint_path ... --task_data_config_path ...` (details in `VISUALIZATION_README.md`).

## Coding Style & Naming Conventions
- Python with 4-space indentation; prefer PEP 8 naming (`snake_case` for funcs/vars, `CamelCase` for classes).
- Keep concerns separated: model logic in `models/`, training loops in `exp/`, config changes in `data_provider/data_config/`.
- Linting: `ruff` is available via `environment.yml`; run `ruff check .` before PRs when possible.

## Testing Guidelines
- No formal test suite is included. Add lightweight smoke checks around the touched entrypoint (e.g., a short `--is_training 0` run) and note expected runtime/device.
- If adding tests, co-locate small scripts near the relevant module and document how to run them.

## Commit & Pull Request Guidelines
- Git history is minimal (`init`), so establish good habits now: use concise Conventional Commit prefixes (`feat:`, `fix:`, `docs:`) and mention the dataset/config touched (e.g., `baseline/ROT`).
- PRs: include purpose, key flags/configs, dataset, checkpoints/log locations, and any visualization outputs. Link issues/experiments where relevant.

## Configuration & Artifacts
- Do not commit large artifacts (datasets, `checkpoints/`, `logs/`, generated figures). Extend `.gitignore` if new output paths are introduced.
- Keep YAML configs reproducible; commit config changes alongside code changes that depend on them.
