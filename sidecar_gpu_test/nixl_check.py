import importlib
import os

try:
    import nixl
    print("nixl:", getattr(nixl, "__file__", "?"))
except Exception as e:
    print("nixl import FAILED:", e)

try:
    from nixl._api import nixl_agent  # noqa: F401
    print("nixl_agent OK")
except Exception as e:
    print("nixl_agent FAILED:", e)

import vllm
base = os.path.dirname(vllm.__file__)
print("vllm:", base)
for root, _dirs, files in os.walk(os.path.join(base, "distributed", "kv_transfer")):
    for f in files:
        if "nixl" in f.lower():
            print("found:", os.path.join(root, f))

for mod in [
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector",
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlConnector",
]:
    try:
        importlib.import_module(mod)
        print("import OK:", mod)
    except Exception as e:
        print("import FAIL:", mod, "->", type(e).__name__, e)
