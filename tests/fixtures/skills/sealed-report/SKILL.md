---
name: sealed-report
description: Use when a test needs a built-in Skill that declares an attestation, a sealed artifact, a kit and required reads.
attestation: scripts/case_validate.py
sealed_artifact:
  name: case.json
  gate: scripts/verify_sealed_case.py
  unsealed_label: "Unsealed research memo / 未封存研究备忘"
kit: scripts
required_reads:
  - path: references/gate-grounding.md
    when: before the first number — identity, as-of and provenance
  - path: references/gate-audit.md
    when: before delivery — the rule table and the instant-fail list
---

# Sealed report

A neutral Skill for the tests of the Skill machinery. It stands in for a shipped Skill that
builds a sealed document: its hub declares the verifier that attests the document, the gate that
checks the sealed artifact, the kit of scripts the Skill materializes, and the references a Run
must read before it starts and before it delivers.

It is not a built-in Skill. Tests point a catalog at the copy `tests/support/skills.py` builds, which
adds `scripts/engine.py`: a script too large to check in, written when the tests run.

## Files

| Path | Read it when |
|---|---|
| `references/gate-grounding.md` | Before the first number |
| `references/gate-audit.md` | Before delivery |
| `references/gotchas.md` | A refusal needs explaining |
| `scripts/engine.py` | Never by hand; materialize the kit |
| `scripts/case_validate.py` | Validate and seal the document |
