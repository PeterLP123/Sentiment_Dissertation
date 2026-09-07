# ARS author-adjudication request

Status: **AWAITING EXPLICIT AUTHOR CONFIRMATION**. This is not an `author-adjudication/1.0` sidecar.

- Roadmap SHA-256: `21c899169d4aad8d516d23a51c0ec98f829c5f4174bfe17582f474b2728d3ff9`
- Claim-surface manifest SHA-256: `9f0e6b995243ca4ca23bb27cdfd4154bb62a1d9e676fd05442736bfe6cedc970`
- Proposed triage for every item: `will_address`
- Proposed operation for every listed block: `replace_block`

## Exact proposed authority

| Item | Blocks | Operation |
|---|---|---|
| `REV-ESTIMAND-ALIGNMENT` | `B0004, B0006, B0012, B0014, B0016` | `replace_block` |
| `REV-INTERPRETATION-BOUNDARY` | `B0004, B0008, B0010, B0014, B0016` | `replace_block` |
| `REV-HYPOTHESIS-LANGUAGE` | `B0006, B0014, B0016` | `replace_block` |
| `REV-TEMPORAL-LITERATURE` | `B0008, B0036` | `replace_block` |
| `REV-WORD-COUNT` | `B0004, B0006, B0008, B0010, B0012, B0014, B0016` | `replace_block` |
| `REV-PRESENTATION` | `B0004, B0038` | `replace_block` |

## Changed canonical surfaces

- `__binary_artifact_manifest__`
- `manuscript/chapters/01_introduction.tex`
- `manuscript/chapters/02_literature.tex`
- `manuscript/chapters/03_data.tex`
- `manuscript/chapters/04_methodology.tex`
- `manuscript/chapters/05_results.tex`
- `manuscript/chapters/06_conclusion.tex`
- `manuscript/main.tex`
- `manuscript/references.bib`

To authorize the deterministic sidecar and patch application, the author must explicitly confirm this exact roadmap hash and every listed block/operation mapping.
