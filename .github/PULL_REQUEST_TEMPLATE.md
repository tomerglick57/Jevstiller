**What this changes**

**Why**

**Checks**
- [ ] `pytest -q` and `ruff check .` pass
- [ ] If routing, calibration or the audit channel changed: DESIGN.md §5 / §7.8 still hold (thresholds from a bound, never a point estimate; deferred rows never enter calibration)
- [ ] If the public API changed: `python tests/test_public_api.py --update` and a CHANGELOG entry
