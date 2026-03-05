# mlops-cicd-pipleine

ML project with train/evaluate/inference and a full CI/CD pipeline (PR checks + staging deploy with rollback).

## Project layout

- **`src/`** – Model code: `train.py`, `evaluate.py`, `data.py`, `model.py`, `inference/app.py`
- **`tests/`** – Unit and data-processing tests
- **`scripts/compare_metrics.py`** – Metric gating (compare new metrics to baseline)
- **`baselines/metrics.json`** – Stored baseline metrics for gating
- **`Dockerfile`** – Inference image (slim, no training deps)

## Quick start

```bash
# Create venv and install
python -m venv .venv && .venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -e ".[dev]"

# Train and evaluate
python -m src.train --out-dir artifacts
python -m src.evaluate --model-path artifacts/model.joblib --out metrics.json

# Run tests
pytest tests -v
ruff check src tests && black --check src tests

# Compare metrics to baseline (gate)
python scripts/compare_metrics.py --new metrics.json --baseline baselines/metrics.json

# Run inference API (with model)
# Command Prompt (cmd):
set MODEL_PATH=artifacts\model.joblib
uvicorn src.inference.app:app --host 0.0.0.0 --port 8000
# PowerShell: $env:MODEL_PATH = "artifacts/model.joblib"
# Then: GET http://localhost:8000/health , POST http://localhost:8000/predict with {"features": [[...20 floats...]]}
```

---

## CI/CD

### Overview

- **PR workflow** (`.github/workflows/pr.yml`): runs on every pull request. Lint, type check, unit tests, data/feature tests, lightweight training + evaluation, and metric gating against `baselines/metrics.json`.
- **Deploy workflow** (`.github/workflows/deploy-staging.yml`): runs on push to `master` or manual dispatch. Builds the inference Docker image, pushes to the registry, deploys to the **staging** environment, runs post-deploy smoke tests, and **automatically rolls back** if those tests fail.

### PR workflow jobs

| Job          | What it does |
|-------------|--------------|
| **lint**    | `ruff check` and `black --check` on `src`, `tests`, `scripts`. |
| **typecheck** | `mypy src` (currently optional; remove `\|\| true` in the workflow to fail on type errors). |
| **test**    | `pytest tests` with coverage. |
| **data-tests** | Pytest on `tests/test_data.py` (feature validation, data loading). |
| **eval-gate** | Runs `python -m src.train`, then `python -m src.evaluate`, then `scripts/compare_metrics.py`. Fails if accuracy or F1 drops by more than 0.5% vs baseline. |

Pip dependencies are cached by `pyproject.toml` hash to speed up runs.

### Deploy workflow jobs

| Job                 | What it does |
|---------------------|--------------|
| **build-and-push**  | Builds the inference Dockerfile, tags image with git SHA, pushes to GitHub Container Registry (GHCR). |
| **deploy**          | Uses `kubectl` to set the staging deployment image to the new tag and waits for rollout. |
| **post-deploy-tests** | Calls `/health` and `/predict` on the staging base URL. Fails if health is not 200 or predict returns an unexpected status. |
| **rollback**        | Runs only if **deploy** succeeded but **post-deploy-tests** failed. Runs `kubectl rollout undo` to revert to the previous revision. |

Rollback does **not** need a stored “previous tag”: Kubernetes keeps rollout history, so `kubectl rollout undo` restores the prior image. On the first deploy there is no previous revision; the rollback job then no-ops gracefully.

### Required secrets and environment

Use GitHub **Environments**: create an environment named **`staging`** and set:

| Secret / config   | Required | Description |
|--------------------|----------|-------------|
| **KUBE_CONFIG**    | Yes      | Base64-encoded kubeconfig so the runner can talk to your cluster (e.g. `cat ~/.kube/config \| base64 -w0`). |
| **STAGING_BASE_URL** | Yes (for smoke tests) | Base URL of the staging app (e.g. `https://staging.example.com`). Used for `/health` and `/predict`. If unset, HTTP smoke tests are skipped and the job still passes. |

For **GHCR** (default), the workflow uses `GITHUB_TOKEN`; no extra registry secrets are needed. To use another registry, set `REGISTRY_URL`, `REGISTRY_USERNAME`, `REGISTRY_PASSWORD` and adjust the workflow to use them.

### Staging Kubernetes setup

Ensure your cluster has:

1. **Namespace**: `staging`
2. **Deployment**: name `staging-app`, container name `app`, with an initial image (e.g. the first time use the same image as the first workflow run).

Example (adjust image and resources as needed):

```bash
kubectl create namespace staging
kubectl create deployment staging-app -n staging --image=ghcr.io/YOUR_ORG/mlops-cicd-pipleine:latest
# Or apply a YAML that sets imagePullSecrets if the image is private.
```

After that, the workflow will update the deployment image to `ghcr.io/<repo>:<git-sha>` on each run.

### How to update baseline metrics

When you intentionally improve the model and want to allow that in PRs:

1. Merge or run on master so you have a green eval.
2. Copy the new metrics into the repo:  
   `cp metrics.json baselines/metrics.json` (or from CI artifact).
3. Commit and push `baselines/metrics.json`.

PRs will then gate against this new baseline. Thresholds are configurable in the compare script and in the workflow:

```bash
python scripts/compare_metrics.py --new metrics.json --baseline baselines/metrics.json \
  --accuracy-threshold 0.01 --f1-threshold 0.01
```

### How rollback works

- **Trigger**: Rollback runs only when **deploy** succeeded and **post-deploy-tests** failed (or a later step that runs after deploy fails).
- **Action**: The workflow runs `kubectl rollout undo deployment/staging-app -n staging`. Kubernetes reverts the deployment to the previous revision (previous image tag). No need to pass the “previous” tag; the cluster stores rollout history.
- **First deploy**: On the first deploy there is no previous revision; the rollback step detects this and exits successfully without failing the workflow.

---

## Running locally and testing rollback

### Run everything locally

One-time: `pip install -e ".[dev]"`. Then run the full PR-style checks (lint, tests, train, evaluate, metric gate):

**Command Prompt (cmd)** — run each line, or paste the block:

```cmd
ruff check src tests scripts
black --check src tests
pytest tests -v --cov=src
python -m src.train --out-dir artifacts
python -m src.evaluate --model-path artifacts/model.joblib --out metrics.json
python scripts/compare_metrics.py --new metrics.json --baseline baselines/metrics.json
```

Optional: `mypy src` for type checking. If all commands succeed, your branch matches what the PR workflow runs.

Run the API with a trained model:

```cmd
set MODEL_PATH=artifacts\model.joblib
uvicorn src.inference.app:app --host 0.0.0.0 --port 8000
```
Then in another terminal: `curl http://localhost:8000/health` and `curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"features\": [[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]]}"

### Test rollback (with a real cluster)

1. Set up the `staging` environment and `KUBE_CONFIG`, `STAGING_BASE_URL`.
2. Merge to master once so a first revision is deployed.
3. Intentionally break post-deploy (e.g. temporarily point `STAGING_BASE_URL` to a URL that returns 503, or break the app so `/health` fails).
4. Push to master or run the deploy workflow manually. The deploy job will succeed; post-deploy-tests will fail; the rollback job will run and execute `kubectl rollout undo`, restoring the previous working revision.
5. Fix the URL or app and re-run; verify the new revision is live again.

### Docker (inference only)

```bash
docker build -t my-inference:local .
docker run -p 8000:8000 -e MODEL_PATH=/data/model.joblib -v "%CD%\artifacts\model.joblib:/data/model.joblib" my-inference:local
```

Then hit `http://localhost:8000/health` and `http://localhost:8000/predict` as above.
