# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Nothing a DlightRAG process runs reports to a telemetry backend.

The ONNX runtime that the conversion stack loads carries Microsoft's 1DS telemetry, on by
default in the official wheels: it uploads the runtime's model metadata and this machine's
identifiers, and its uploader thread raced the runtime's own teardown and aborted processes.
The switch is read while the runtime initialises, so it has to be on before the first import.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[2]

# Clear anything inherited, then watch what the variable holds the moment the runtime arrives.
_PROBE = """
import json, os, sys
os.environ.pop("ORT_DISABLE_TELEMETRY", None)
seen = []
def watch(event, args):
    if event == "import" and args[0] == "onnxruntime":
        seen.append(os.environ.get("ORT_DISABLE_TELEMETRY"))
sys.addaudithook(watch)
from dlightrag.engine.answer.resources import converters
converters.disable_model_telemetry()
import onnxruntime
print(json.dumps(seen))
"""


def test_the_switch_is_on_when_the_runtime_is_first_imported() -> None:
    probe = subprocess.run(  # noqa: S603 - the interpreter running this suite, with a fixed program
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, check=True
    )
    assert json.loads(probe.stdout.strip()) == ["1"]


def test_every_service_that_shares_the_image_opts_out() -> None:
    services = yaml.safe_load((_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))[
        "services"
    ]
    sharing = {
        name: service
        for name, service in services.items()
        if service.get("image") == "dlightrag:local"
    }
    assert sharing, "the compose file names the image its services share"
    missing = sorted(
        name
        for name, service in sharing.items()
        if str((service.get("environment") or {}).get("ORT_DISABLE_TELEMETRY")) != "1"
    )
    assert missing == [], f"a service shares the image and would report to Microsoft: {missing}"
