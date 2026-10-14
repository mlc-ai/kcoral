#!/usr/bin/env bash
# Replace every libhsa-runtime64 the server can load (/opt/rocm and the copy
# bundled with the PyTorch wheel) with the patched build, keeping backups.
set -euo pipefail
prefix="$1"
built=$(ls "$prefix"/lib*/libhsa-runtime64.so.*.* 2>/dev/null | head -1)
[ -n "$built" ] || { echo "no patched libhsa-runtime64 under $prefix" >&2; exit 1; }
echo "patched build: $built"

torch_lib=$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')
targets=()
for dir in /opt/rocm/lib "$torch_lib" /opt/venv/lib/python3*/site-packages/_rocm_sdk_core/lib; do
  [ -d "$dir" ] || continue
  for f in "$dir"/libhsa-runtime64.so*; do
    [ -e "$f" ] && [ ! -L "$f" ] && targets+=("$f")
  done
done
[ ${#targets[@]} -gt 0 ] || { echo "no libhsa-runtime64 found to replace" >&2; exit 1; }
for f in "${targets[@]}"; do
  cp -a "$f" "$f.orig"
  cp "$built" "$f"
  echo "replaced $f (backup: $f.orig)"
done
