#!/usr/bin/env bash
# =============================================================================
# scripts/setup_gpu_env.sh
# =============================================================================
# Reinstalls PyTorch with CUDA 12.1 support in ja_scraper/venv.
# Run this once before any training session on protouno.
#
# Usage:
#   cd /mnt/storage/projects/DLCGIPG
#   bash scripts/setup_gpu_env.sh
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${PROJECT_ROOT}/ja_scraper/venv"
PYTHON="${VENV}/bin/python"
PIP="${VENV}/bin/pip"

echo "============================================================"
echo " DLCGIPG — GPU environment setup"
echo " Project root : ${PROJECT_ROOT}"
echo " Venv         : ${VENV}"
echo "============================================================"

# ── activate venv ─────────────────────────────────────────────────────────────
if [[ ! -f "${PYTHON}" ]]; then
    echo "ERROR: venv not found at ${VENV}"
    echo "       Create it first: python3 -m venv ja_scraper/venv"
    exit 1
fi

source "${VENV}/bin/activate"
echo "✓ Activated venv: $(python --version)"

# ── uninstall CPU torch if present ────────────────────────────────────────────
echo ""
echo "Removing existing CPU-only torch / torchvision (if present) …"
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# ── install GPU torch (CUDA 12.1) ─────────────────────────────────────────────
echo ""
echo "Installing torch + torchvision (cu121) …"
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121 \
    --upgrade

# ── install training dependencies ─────────────────────────────────────────────
echo ""
echo "Ensuring training dependencies are installed …"
pip install \
    numpy \
    pandas \
    scikit-learn \
    Pillow \
    tqdm \
    --quiet

# ── verify CUDA ───────────────────────────────────────────────────────────────
echo ""
echo "Verifying CUDA availability …"
python - <<'EOF'
import torch
print(f"  torch version  : {torch.__version__}")
print(f"  CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU name       : {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version   : {torch.version.cuda}")
    print(f"  Compute cap.   : {torch.cuda.get_device_capability(0)}")
    # quick smoke-test: move a tensor to GPU
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x.T
    print(f"  Smoke test     : matmul on GPU — OK ({y.shape})")
else:
    print("  ⚠  No GPU detected. Check driver / CUDA install.")
EOF

echo ""
echo "============================================================"
echo " Setup complete."
echo " To activate in new shell:"
echo "   source ${VENV}/bin/activate"
echo "============================================================"
