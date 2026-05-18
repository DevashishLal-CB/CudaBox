#!/bin/bash

set -eo pipefail

SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]:-$0}")/")"
source "$SCRIPT_DIR/../env.sh"

VENV_PATH="$HOME/uv_venv/cudabox"
if [ ! -d "$VENV_PATH" ]; then
  bold_status "CREATING UV VENV AT $VENV_PATH" "green"
  uv venv "$VENV_PATH"
fi
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

bold_status "INSTALLING BUILD DEPENDENCIES" "green"
uv pip install "scikit-build-core>=0.11" wheel "torch>=2.7.0" triton numpy pre-commit pytest

# Install pre-commit git hooks if a config exists and hooks aren't installed yet.
REPO_ROOT="$(realpath "$SCRIPT_DIR/../..")"
if [ -f "$REPO_ROOT/.pre-commit-config.yaml" ] && \
   [ ! -f "$REPO_ROOT/.git/hooks/pre-commit" ]; then
  bold_status "INSTALLING PRE-COMMIT HOOKS" "green"
  (cd "$REPO_ROOT" && pre-commit install)
fi

bold_status "BUILDING CUDABOX" "green"
uv build --wheel -Cbuild-dir=build . --verbose --color=always \
  --no-build-isolation --config-settings=cmake.build-type="Debug"
bold_status "BUILD COMPLETE" "green"
ls dist

bold_status "INSTALLING CUDABOX" "green"
uv pip install ./dist/cudabox*.whl --force-reinstall
bold_status "INSTALL COMPLETE" "green"

bold_status "VERIFYING CUDABOX IMPORT + LISTING REGISTERED TORCH OPS" "green"
python - <<'PYEOF'
import importlib
import sys

import torch

mod = importlib.import_module("cudabox")
print(f"Imported cudabox from: {mod.__file__}")

ns = "cudabox"
all_ops = torch._C._dispatch_get_all_op_names()
ns_ops = sorted(name for name in all_ops if name.startswith(f"{ns}::"))

if not ns_ops:
    print(f"ERROR: no torch ops registered under namespace '{ns}'", file=sys.stderr)
    sys.exit(1)

print(f"Registered torch ops under torch.ops.{ns} ({len(ns_ops)}):")
for op in ns_ops:
    print(f"  - torch.ops.{op.replace('::', '.')}")
PYEOF

# If this script was sourced, the activation persists in the caller's shell.
# Otherwise, print the command to activate manually.
(return 0 2>/dev/null) || \
  bold_status "TO ACTIVATE VENV: source $VENV_PATH/bin/activate" "yellow"
