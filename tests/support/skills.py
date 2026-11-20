# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The neutral built-in Skill the tests of the Skill machinery run on.

Sealing, attestation, required reads, kits and paging are properties of any Skill that declares
them. ``tests/fixtures/skills/sealed-report`` is one that declares all of them, so those tests do
not depend on what a shipped Skill happens to contain.

The fixture directory holds the small files. The Skill's engine, a script over 500,000 bytes that a
model profile has to page, is written here at test time into a copy of the directory
(:func:`sealed_skills_root`), so no generated file is checked in.
"""

import atexit
import functools
import shutil
import tempfile
from pathlib import Path

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "skills"
SEALED_SKILL = "sealed-report"
SEALED_ENGINE = "scripts/engine.py"
#: A line of the engine that its tests look for: a label the engine writes, which no reference
#: carries, so reader masking covers it and the publication prose index leaves it out.
SEALED_ENGINE_LABEL = (
    "a SIMULATED tail, not a measured one: it is the generator's own tail, "
    "not a frequency measured on this sample"
)
_ENGINE_FUNCTIONS = 8600  # 61 bytes each: the file is over the 500,000 bytes the paging tests need


def _engine_source() -> str:
    """The engine: a docstring, one label, then enough small functions to need paging.

    The functions are written the way ``ruff format`` leaves them, so the text is stable.
    """
    functions = "\n\n".join(
        f"def step_{number:05d}(value: int) -> int:\n    return value + {number}\n"
        for number in range(1, _ENGINE_FUNCTIONS + 1)
    )
    return (
        '"""A stand-in for a large audited engine: one label, then functions that need paging."""\n'
        f'\nLABEL = "{SEALED_ENGINE_LABEL}"\n\n\n{functions}'
    )


@functools.cache
def sealed_skills_root() -> Path:
    """A ``builtin_root`` that serves the neutral Skill alone, engine included.

    Built once per test process in a temporary directory that is removed when the process exits.
    """
    root = Path(tempfile.mkdtemp(prefix="sealed-skills-"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    shutil.copytree(_FIXTURES, root, dirs_exist_ok=True)
    (root / SEALED_SKILL / SEALED_ENGINE).write_text(_engine_source(), encoding="utf-8")
    return root
