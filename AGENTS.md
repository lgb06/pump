# Repository Guidelines

## Project Structure & Module Organization

- `run.py`, `run_pretrain.py`, `run_fewshot.py`, `run_DG.py`: primary entrypoints for supervised, pretrain, few-shot, and domain-generalization runs.
- `exp/`: experiment drivers (e.g., `exp_sup.py`, `exp_pretrain.py`) that wire data, model, training, and evaluation.
- `models/`: model definitions (e.g., `RmGPT.py`, baselines like `ResNet1D.py`); shared building blocks live under `models/layers/`.
- `data_provider/`: dataloaders and factories plus YAML configs in `data_provider/data_config/**` (e.g., `baseline/ROT.yaml`).
- Outputs: checkpoints under `checkpoints/` and logs under `logs/` (both are gitignored). Visualization artifacts default to `visualization_results/`.

## Build, Test, and Development Commands

- Create environment: `conda env create -f environment.yml` then `conda activate time_series` (Python 3.9).
- Train (supervised): `python run.py --is_training 1 --task_data_config_path data_provider/data_config/baseline/ROT.yaml`.
- Evaluate: `python run.py --is_training 0 --pretrained_weight checkpoints/<exp>/checkpoint.pth`.
- Visualizations: add `--visualize` (and optionally `--extract_attention`) or run `python visualize_analysis.py --checkpoint_path ... --task_data_config_path ...` (see `VISUALIZATION_README.md`).
- Batch scripts (Linux/WSL): `bash run_and_log.sh ./scripts/supervised_learning/RmGPT_pump_supervised.sh` (writes `logs/<script_dir>/...`).

## Coding Style & Naming Conventions

- Python: 4-space indentation; prefer PEP 8 naming (`snake_case` for functions/vars, `CamelCase` for classes).
- Keep changes localized: model code in `models/`, training logic in `exp/`, and data/config changes in `data_provider/`.
- Linting: the environment includes `ruff`; run `ruff check .` before submitting if available.

## Testing Guidelines

- This repo is script-driven and does not ship a dedicated test suite; prefer adding small, runnable sanity checks near the affected entrypoint (e.g., a minimal `--is_training 0` smoke run).

## Commit & Pull Request Guidelines

- Git history is not present in this snapshot (no `.git`), so there is no established commit-message convention to mirror.
- Recommended: use Conventional Commits (e.g., `fix:`, `feat:`, `docs:`), include the dataset/config touched (e.g., `baseline/ROT`), and link issues/experiments in the PR description.

## Configuration & Artifacts

- Do not commit large artifacts (datasets, `checkpoints/`, `logs/`, generated figures). If you add new output paths, extend `.gitignore` accordingly.
- Keep YAML configs reproducible: commit config changes alongside the code that consumes them.

