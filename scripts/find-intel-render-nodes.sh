#!/usr/bin/env bash
# Map Intel GPUs to /dev/dri render nodes and by-path names.
# Run on the GPU host. Do not pass all of /dev/dri into the container.
set -euo pipefail

echo "=== PCI display / 3D devices ==="
if command -v lspci >/dev/null 2>&1; then
  lspci -Dnn | grep -Ei 'VGA|Display|3D|Graphics' || true
else
  echo "lspci not installed"
fi

echo
echo "=== /dev/dri ==="
ls -l /dev/dri 2>/dev/null || echo "no /dev/dri"

echo
echo "=== /dev/dri/by-path (Level Zero enumerates through these) ==="
if [ -d /dev/dri/by-path ]; then
  ls -l /dev/dri/by-path
  echo
  echo "Copy DRI_RENDER_* and DRI_BYPATH_* from the pci-*-render lines for the B70s."
  echo "Example: pci-0000:17:00.0-render -> ../renderD129"
  echo
  echo "Render node numbers can change after a reboot; PCI addresses do not."
  echo "Recheck DRI_RENDER_* against this listing, or run:"
  echo "  ./scripts/resolve-dri-from-bypath.sh --write .env"
else
  echo "no /dev/dri/by-path"
fi

echo
if command -v xpu-smi >/dev/null 2>&1; then
  echo "=== xpu-smi discovery ==="
  xpu-smi discovery || true
else
  echo "xpu-smi not on PATH (optional)"
fi
