#!/bin/bash
set -euo pipefail

SITE_PACKAGES="/usr/local/lib/python3.12/dist-packages"
MODELOPT="$SITE_PACKAGES/vllm/model_executor/layers/quantization/mxfp4.py"

cd "$SITE_PACKAGES"

echo "[fix-ds4-gpu-cache]  Patching GPU empty cache"

python3 - <<'PY'
from pathlib import Path

path = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/mxfp4.py')
text = path.read_text()
orig = text

old = '''        if self.mxfp4_backend == Mxfp4MoeBackend.NONE:
            return

        self._setup_kernel(layer, w13, w2, w13_scale, w2_scale, w13_bias, w2_bias)
'''
new = '''        if self.mxfp4_backend == Mxfp4MoeBackend.NONE:
            return

        self._setup_kernel(layer, w13, w2, w13_scale, w2_scale, w13_bias, w2_bias)
        del w13, w2, w13_scale, w2_scale, w13_bias, w2_bias
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-ds4-gpu-cache] ERROR: no available entry point found in mxfp4.py')
    text = text.replace(old, new)
else:
    print('[fix-ds4-gpu-cache] cache empty mechanism found')

if text != orig:
    path.write_text(text)
    print('[fix-ds4-gpu-cache]  patched', path)
else:
    print('[fix-ds4-gpu-cache]  no changes needed')
PY

# Clear stale bytecode for the patched module.
find "$SITE_PACKAGES/vllm/model_executor/layers/quantization" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

python3 -m py_compile "$MODELOPT"
