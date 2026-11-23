# SOL CPU integration tests

This environment runs the SOL unit and data-pipeline integration tests on Linux
x86_64 with Python 3.12. It uses the official PyTorch CPU wheel index and PyPI
for the remaining packages. The Docker Python SDK is installed because importing
the production environment package imports its Docker backend; the tests do not
start or require a Docker daemon.

The pinned OLMo-core Git commit is kept in requirements.in. Do not install the
repository's root dependency groups for this test environment: those include
GPU-oriented packages such as vLLM and CUDA-specific Torch sources.

## Create and install the environment

Run these commands on Linux x86_64 from the repository root:

```bash
cd tests/sol_cpu
uv venv --python 3.12 .venv
uv pip compile \
  --no-config \
  --python .venv/bin/python \
  --python-platform x86_64-unknown-linux-gnu \
  --torch-backend=cpu \
  --generate-hashes \
  requirements.in \
  -o requirements.lock
uv pip sync \
  --no-config \
  --python .venv/bin/python \
  --torch-backend=cpu \
  requirements.lock
```

Both commands pass `--torch-backend=cpu` explicitly because `--no-config`
intentionally ignores `uv.toml`. This selects the official CPU wheel for Torch
while other packages use the default PyPI index. The compile command targets
Linux x86_64 even if it is run from another platform; installation and tests
should still be performed on Linux x86_64.

## Run the SOL tests

From the repository root:

```bash
CUDA_VISIBLE_DEVICES="" tests/sol_cpu/.venv/bin/python -m pytest -q \
  open_instruct/test_sol_utils.py \
  open_instruct/test_sol_reward_config.py \
  open_instruct/test_sol_utils_integration.py
```

The empty `CUDA_VISIBLE_DEVICES` prevents these CPU tests from selecting a
visible GPU. No training, distributed worker launch, or Docker container is
started by this test command.
