# Runtime pins

Stage 1 uses Python 3.11, PyYAML 6.0.3, pytest 7.4.0, and the OpenAI
Python SDK 2.8.1. ALFWorld 0.4.2 and TextWorld 1.7.0 are optional and
were located locally. These are direct dependency pins, not a claim of a
portable transitive lock. The local environment inventory is recorded in
`reports/stage1_environment.json`; every experiment command records its runtime.
No packages, model weights, or data are downloaded by tests. No torch,
Transformers, or GPU runtime is required by this stage.

For a fresh environment, install the requirements files when network access
is authorized, then `pip install --no-deps --no-build-isolation -e .`.
Offline, `PYTHONPATH=src:.` also runs all entry points from the checkout.
The ALFWorld virtual environment can be used directly without modifying it;
see the local validation report for its test invocation.

Student download/license/template checks and training dependency pins are
future work. The configured student revision is user-specified; Stage 1
does not claim to have downloaded or trained it. The teacher canary is the
user-reported result, not an API call performed by this codebase setup.

Stage 2a adds direct client pin `httpx==0.28.1` (already installed as an OpenAI
SDK dependency). The serving runtime was inspected read-only at vLLM 0.27.1 /
Transformers 5.14.1 in its separate venv. Serving dependencies are deliberately
not imported by environment workers or installed into the project venv. No
packages were installed for Stage 2a.

Stage 2b uses the existing training environments without installation:
HPG Python (`TRAIN_PYTHON` in `configs/hpg.env`) has torch 2.11.0+cu128,
Transformers 5.14.1, PEFT 0.20.0, Accelerate 1.14.0 (user-provided inventory).
The RAI reference venv (`RAI_REFERENCE_PYTHON`) has Python 3.12 and torch
2.13.0+cu130 with the same Transformers/PEFT/Accelerate. The project `.venv`
has Python 3.11 and no torch; tokenizer/model tests therefore run separately
in the RAI reference venv with CUDA hidden. Optional `training` dependencies
pin the HF libraries; select the already-approved machine-specific torch build
separately. No download or installation is performed by training or tests.
