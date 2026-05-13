# Gate 2 Known Risks

## Deferred Risks

### C-2 — Mutable ScoredSignal / TieredSignal
Deferred until broader immutability migration.

### C-3 — DailyCounter concurrency atomicity
Relevant only after concurrent multi-symbol scanning is introduced.

### H-2 — is_hostile semantic divergence
Planned cleanup during gate abstraction pass.

## Notes

These risks are acknowledged and intentionally deferred
to avoid destabilizing the hardened Gate 2 architecture.

Current priority:
- architecture stability
- invariant enforcement
- telemetry correctness
- pipeline safety