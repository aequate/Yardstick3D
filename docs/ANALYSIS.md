# Analysis notes

Two things that shaped how results in this repo are reported: a decomposition of where scale error comes from, and a
running ledger of which claims the evidence supports.

## Where the error comes from

Grounded error is limited from three directions:

1. **Geometry.** Even the best possible single scale leaves some error, because the model's camera path has the
   wrong shape. That error is the Sim(3) floor: ATE after fitting rotation, translation *and* scale against the
   reference. No scale estimator can beat it.
2. **The cue.** The GNSS path length `Lg` over a window is not exactly the true path length `Lt`. The receiver's
   noise, a phone's position smoothing or a clock offset all show up here.
3. **The estimator.** How the solver turns per-interval distances into one scale (`s_hat`), compared with simply
   dividing the cue path by the model's path (`s_cue = Lg / Lv`).

For each window these combine exactly as a product of scale ratios:

```
s_hat / s_opt  =  (s_hat / s_cue)  ×  (Lg / Lt)  ×  (s_path / s_opt)
                  estimator          cue           path vs ATE-optimal scale
```

where `s_opt` is the ATE-optimal scale and `s_path = Lt / Lv`. The identity holds per window, not for medians across
windows, and it only covers the scale. ATE itself does not split into additive parts, so no "x m from the cue,
y m from geometry" breakdown is claimed. The helpers are in
[`yardstick3d/evaluation/three_limits.py`](../yardstick3d/evaluation/three_limits.py).

Two descriptive ratios are reported but never used as evidence on their own:

- **leverage** = (raw − floor) / raw: how much there was to gain from scaling. It depends on the arbitrary scale the
  model happens to output, so a high value is close to automatic.
- **capture** = (raw − grounded) / (raw − floor): how much of that gain grounding achieved. Near 1 whenever the raw
  scale is badly off, so it is also not a headline number.

The number that matters is grounded ATE next to the Sim(3) floor. On every pre-registered drone window it landed
within 1.00–1.10× of the floor, so what remains is the model's geometry, which GNSS cannot fix.

What this showed in practice:

- On ADVIO the phone's location track ran about 7% short of the true path on two sequences and about 21% long on
  another, so the cue, not the solver, set the scale error there. The u-blox receivers on the UrbanNav car and the
  drones were close to exact (cue/reference path ratio 0.996 on UrbanNav).
- On ADVIO-22/23 and the UrbanNav drive the floor itself is several metres; grounding gets close to it but the
  result is still metres off.
- COLMAP's floor at full frame rate (3.49 m) is far above the foundation models' (0.31–0.57 m), which is why the
  same cue grounds it much worse.

## Claim ledger

Every experiment ended with a written decision on what it supports, and the earlier decisions were never edited.
Before any result was promoted, two adversarial reviews were run against it: one checking integrity (hashes, data
leakage, whether anything was rescored) and one arguing against the proposed interpretation.

| Date | Experiment | Decision |
|---|---|---|
| 2026-09-08 → 13 | ADVIO, retrospective, DA3 + VGGT + COLMAP | Characterisation only. Sequences had been looked at during development. |
| 2026-09-12 | UrbanNav-HK, pre-registered, DA3-BASE | Grounded close to the floor, but one model only: not promoted. |
| 2026-09-29 | MARS-LVIG airfield GNSS01, pre-registered, DA3 + VGGT | **FAIL.** The u-blox timestamps were GPS time, 18 s off UTC, and the time-alignment gate caught it. The numbers looked excellent (0.49 m and 0.24 m) and the result still stands as a failure; it was never rescored. |
| 2026-09-30 | Island contracts v2–v3 | Stopped before any windows, inference or reference were opened: the flights failed the contract's ground-truth-free timing checks. The design was revised into v5. |
| 2026-09-30 | MARS-LVIG island (contract v5), DA3 + VGGT | Conditional pass, with 11 written caveats (small sample, metre-level accuracy, naive solver only, no model ranking, …). |
| 2026-10-01 | MARS-LVIG airfield GNSS02/03 (R1), DA3 + VGGT | Replicated at a second site, on flights not used before. |

What the evidence does **not** support, and the repo does not claim:

- that sparse GNSS fixes the scale everywhere (poor-geometry windows stay at 2–4 m even with every measurement);
- detecting, without ground truth, when a reconstruction is too broken to ground (no detector tried worked reliably);
- that one measurement is always enough (it depends on the window);
- a calibrated confidence bound on the scale (the maths exists but no cue had a calibrated noise model);
- a new estimator: the scale ratio is classical, the contribution is the measurement;
- that one model beats another: VGGT and DA3 both working is replication, not a ranking.
