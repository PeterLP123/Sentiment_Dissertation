# ARS Re-review Report

**Round:** `ars-re-review-2026-08-14-round-2`
**Roadmap:** `8d9412dbb29253b0bf2dc2cb30e81b5e3593f07785afbc6461c124cccd0737d1`
**Decision:** **Accept**
**Checker result:** `re-review synthesis ok`
**Apply-chain witness:** `pass`

## Decision basis

The authorised two-operation patch applied cleanly to canonical blocks `B0014`
and `B0018`, preserving 36 of 38 blocks byte-identically and making no
structural change. Both Round-2 `should_fix` items are fully addressed. Phase
2A recorded no new issue, dissent or escalation exception; Phase 2B verified
both rows and derived **Accept**.

| Roadmap item | Class | Authorised blocks | Phase 2A | Phase 2B | Residual |
|---|---|---|---|---|---|
| `REV-ESTIMAND-ALIGNMENT-R2` | should fix | `B0018` | Fully addressed | Fully addressed | None. The appendix now pairs the frozen `+0.00583` with the matched development baseline `-0.00831` and labels `-0.00914` as separate recent-price-control robustness. |
| `REV-INTERPRETATION-BOUNDARY-R2` | should fix | `B0014`, `B0018` | Fully addressed | Fully addressed | None. Both passages now limit the finding to instability in the measured baseline pipeline association and explicitly leave its cause unidentified. |

The mandatory synthesiser returned:

```text
re-review synthesis ok: round 'ars-re-review-2026-08-14-round-2', revision 1, decision_state 'Accept', apply_chain_witness 'pass'
```

## Verification record

- The final integrity gate is `PASS` with zero open issues. All seven
  research-integrity modes are `CLEAR`.
- Phase E verified all 18 promoted claim families through formal
  `evidence-row/1.0` records, with no distortion or unverifiable claim.
- All 37 bibliography records and all 46 citation-command contexts were
  checked; there are no dangling or ghost citations.
- The originality screen covered 73 of 145 eligible chapter paragraphs plus
  both modified passages and found no suspicious exact or close overlap in 74
  exact-fragment searches.
- The #660 phrase-list carrier correctly records `SNAPSHOT_NOT_PROVIDED`. The
  #672 carrier found no listed inconsistency in its three performed manuscript
  pairs and records the unavailable preregistration pair without laundering it
  into a clean finding.
- Repository validation passed with 64 tests in the existing locked Python
  3.12 environment. Five warnings are third-party `exchange_calendars`
  deprecations. Sealed Notebook 75 was not executed.
- The final 57-page A4 PDF was built and visually inspected on all pages. The
  six numbered chapters contain 10,889 words.

## Evidence-chain hashes

- Original Round-2 draft:
  `5f73b895261c314d22538104b3dfc797b71f06425b3a8554b1df9269466bd7e0`
- Accepted Round-2 draft:
  `d15e6c3762b3637a7a79bd686ff6b6b3f381093688844417779a0d69473599ce`
- Authorised patch:
  `fbce89fe5efd81e088f64026aaba58fd4b820674ba81ae787af34cb68169ca60`
- Continuous Round-1-to-Round-2 evidence bundle:
  `8f09c87b6bf603f8c7471f46525bcbc5c9b14981c2cba20c2baee36b3bcaa467`
- Final claim verification:
  `15091c70c9823f83f1982f74505c0278230788f17e230d10ec5a4d2d53435389`
- Final originality sample:
  `c8c50bb59ce93c1cae45f30776705bc99238127830473be75ab0f4e69be27760`
- Tortured-phrase advisory:
  `677de00e13882b12d9726f48c6dd8f42d1f7ccf8d5e70e9ae850384d16759894`
- Cross-document advisory:
  `c0403b21c8e18078b275499339bcdeb51d1b71ca8ca07fd7b936defa7935156a`
- Final PDF:
  `53ab7883bee6e9dda94107ac74033250023e0edf1d729960746a6aad61d9c078`

## Judge disclosure

Cross-model verification was not configured. This verification round ran on
the same model family that drove the revisions, so over-optimisation to this
judge's latent preferences remains a limitation.
