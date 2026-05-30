# SPDX-License-Identifier: Apache-2.0
"""Load the sidecar ``client`` + generated stubs as a standalone package.

``client.py`` is intentionally dynamo-free, but it lives under the
``dynamo.vllm.sidecar`` namespace whose ``__init__`` imports ``llm_engine``
(which needs ``dynamo._core``, i.e. the built Rust runtime). To CPU-test the
client without dynamo installed, we register a synthetic package whose
``__path__`` points at the sidecar dir and import the pure submodules under it,
bypassing the real package ``__init__``.
"""

from __future__ import annotations

import importlib
import os
import pathlib
import sys
import types

_ALIAS = "_oe_sidecar"
_SIDECAR_DIR = pathlib.Path(
    os.environ.get(
        "OE_SIDECAR_DIR_OVERRIDE",
        str(pathlib.Path(__file__).resolve().parent.parent),
    )
)


def load():
    """Return ``(client_module, pb_module, pb_grpc_module)`` loaded standalone."""
    if _ALIAS not in sys.modules:
        pkg = types.ModuleType(_ALIAS)
        pkg.__path__ = [str(_SIDECAR_DIR)]
        pkg.__package__ = _ALIAS
        sys.modules[_ALIAS] = pkg

    client = importlib.import_module(f"{_ALIAS}.client")
    pb = importlib.import_module(f"{_ALIAS}._openengine.openengine_pb2")
    pb_grpc = importlib.import_module(f"{_ALIAS}._openengine.openengine_pb2_grpc")
    return client, pb, pb_grpc
