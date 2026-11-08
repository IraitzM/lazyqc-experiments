# Family C validation notes (QAOA path / star, B3 vs B5)

Artifacts: `evaluation.json` (7 repeats, base modes),
`evaluation_r15_budget.json` (15 repeats + budget1),
`evaluation_r7_sweep.json` (7 repeats + budgets 8, 32).
Runner: `benchmarks/run_workload_manifest_v2.py --budgets ...`.
All errors ≤ 1e-15 vs independent NumPy oracle. Manifest v1 frozen, untouched.

## Stable findings (all three runs)

- Sharing mechanism intact: B3 → B5 unique tasks 148 → 67 (path) and
  76 → 58 (star), identical to the proposal's historical session.
- Cold joint planning loses on 100% of path/star repeats (path 0.48–0.59×,
  star 0.24–0.27×). Proposal band 0.21–0.59× reproduced.
- Path warm cached joint wins: 1.67× (r15), 1.29–1.50× across runs,
  ≤12% regressions.

## Star warm: parity, not a win

- r7: 1.21× [0.69–1.92]; r15: 1.05× [0.75–1.45]; sweep r7: 1.11× [0.76–1.59].
  Point estimate converges toward ~1.0 as repeats grow; CIs always cross 1.
- Verdict: no warm advantage established on the star. Task reduction 76 → 58
  does not repay into wall-clock here.

## Budget probe (expected_collections = R)

- R=1 falls back to the B3-equivalent plan on both cases (148 / 76 tasks):
  the sharing discovery itself is what the budget cuts.
- Path: R=8 and R=32 both recover the 67-task plan and its warm win
  (1.34× / 1.25×), but cold barely improves (42–43 ms vs 46 ms) because
  incumbent construction, not extra exploration, dominates path search cost.
- Star R=32 sits exactly on the discovery threshold: cold draws mixed
  {58, 76} task counts across repeats.

## Caveat: budget × cache interaction

Star R=32 warm cached locked to the 76-task plan while cold draws mixed
{58, 76}: the shared cache is seeded by whichever plan the first call finds,
so a binding budget makes warm quality seeding-dependent. A reuse-budgeted
policy must either seed the cache with the budgeted search's best-known plan
deterministically or revalidate on hit. Open for Weeks 3–5 cost-model work.

## Implication for the Weeks 3–5 policy

A fixed small R is wrong in both directions: R=1 destroys sharing, and on
the path even R=32 saves almost no cold time. The budget must scale with
expected reuse R (path break-even sat at 18–21 calls in the reuse study),
and marginal-benefit estimation has to price incumbent construction, not
just extra beam exploration.

## Fix 2026-09-24: deterministic reuse budget (amended plan, first task)

Diagnosis: the budget truncated beam exploration on a wall-clock cutoff
(`perf_counter_ns() >= economic_deadline` in `batch_planner.py`), so machine
noise decided which candidates were scored — identical inputs gave {58, 76}
task plans across repeats, and the warm cache locked in whichever plan the
seeding call happened to find.

Change (`src/lazyqc/batch_planner.py`): the allowance (incumbent predicted
cost × R) is now spent as beam expansions, not nanoseconds. Extra states =
min(R × beam_width × outputs, allowance // PLANNING_NS_PER_EXPANSION),
capped by max_search_states; the nominal 1 ms/expansion cost is the order of
the measured rate and is reported for recalibration. Near-zero predicted
savings stop exploration entirely (existing tiny-score test now passes via
the magnitude term, not the clock). Soft wall-clock timeout stays as safety.

Verification (`evaluation_r7_fixed.json`, budgets 1/8/32): all 20
mode/case combinations deterministic (singleton task and state sets).

| Case | R=1 | R=8 | R=32 | Unbudgeted |
|---|---|---|---|---|
| path | 148 tasks / 2 states | 67 / 11 | 67 / 11 | 67 / 11 |
| star | 76 / 1 | 76 / 12 | **58 / 51** | 58 / 78 |

Discovery thresholds mapped: path needs 3–11 expansions, star 13–51.
Star R=32 warm cache now seeds with the 58-task plan — the seeding caveat
above is closed.

## Cost-model v2 instrumentation (begun)

`analysis_stats` now records `incumbent_build_ns` (marginal = planning_ns
minus it), `economic_allowance_ns`, and `economic_state_cap`; the runner
records per-repeat `search_states`/`search_exhausted`. First split probe:

- path joint: incumbent ~221 ms vs marginal ~15 ms (incumbent dominates,
  as the amendment claims; R32: 13.5 + 16.8);
- star joint: incumbent ~46 ms vs marginal ~145 ms (R32: 45.2 + 105.9,
  cap 52 binding).

Caveat: `search_exhausted` conflates beam truncation with cut-proposal
budget exhaustion (path reports exhausted at 11 states via a
"Cut proposal budget exhausted" rejection). v2 reporting should separate
the two flags.

## Weeks 3-5 gate: selection error vs fastest-forced (2026-09-24)

Harness: `benchmarks/evaluate_reuse_policy.py` on recorded deterministic
data (train/test repeat split). Lifetime L(R) = H + R·T from measured cold
planning H and warm cached totals T. Rules: blind B3/B5, deployable
R-threshold (train break-even + declared R), ex-post score-predicted
(measured H + predicted scores).

Family C (path/star, 20 decisions): R-threshold 0 errors; score-predicted
1/20 (star R=512: predicted 0.28 ms saving × 512 marginally beats the H gap
while measured truth disagrees — 109 ms regret). Path train R* = 16,
consistent with the reuse-study 18–21 break-even.

Held-out (`evaluation_heldout.json`: overlapping/disjoint/local-pauli,
30 decisions): score-predicted 10%, R-threshold 13%, always-B3 17%,
always-B5 83%. Disjoint (no sharing possible) is correctly B3 everywhere
by every selective rule. Local-pauli and overlapping flip at R≈128–512,
where ±0.1 ms T noise dominates — boundary selection is timing-noise
limited, a reportable limit of any total-time policy here. The local flip
miss traces to the known ~2× underprediction of execution savings.

## Attacking the execution-prediction gap (2026-09-24)

Per-task diagnosis first (`_one` vs `task_ns` on selected plans): the
task model is actually good — ratios 0.95–1.32, mostly 1.1–1.25, and
reconstruction matches (60–77 µs both sides). The ~2× "saving gap" was
mostly SCOPE, not bias: warm totals include serial scheduling (~3 µs/task
measured independently on family data: 0.46 ms/148, 0.21/67, 0.235/76,
0.166/58 tasks) plus cache-hit planning, which the score excluded.

Fix: `BatchCostModel.scheduling_ns_per_task`, fitted in `calibrate_batch`
through the real `_execute_batch` path on calibration tasks (this session:
4.3 µs/task), added to `price()`; `prediction_scope` reworded.
Per-task width-2 bias (~1.24×) and RY-only calibration mix remain as
second-order follow-ups.

Re-run (`evaluation_r7_sched.json` + `policy_sched.json`): path predicted
saving 1.14 → 1.50 ms (measured experiments+scheduling ≈ 1.7 ms).
Score-predicted stays 1/20 — but the remaining star-R512 "error" dissolved
into measurement noise: train H 66/200 ms vs test 50/244 ms (±20% cold
planning swings), train warm T edge NEGATIVE (−0.9 ms) vs test +0.19 ms.
The oracle itself flips with the repeat block, so the "error" is
unresolvable at 3–4 repeats, not a model failure. Boundary-flip power
analysis (how many repeats resolve a 0.2 ms edge) is the remaining
methodological gap.

Supporting changes: `BatchPlan.beam_truncated` separates beam truncation
from cut-proposal exhaustion; `BatchCostModel.planning_ns_per_expansion`
(calibrated by `calibrate_planning`, fitted ≈ 0.52 ms/expansion, 10×
dispersion across widths) replaces the nominal 1 ms rate in the budget cap,
with old profiles loading via back-compat default.

## Lifetime selector in plan_all (2026-09-24)

`plan_all(..., cutting=CutOptions(..., expected_collections=R), select="lifetime")`
runs independent + joint search deterministically and keeps min predicted
execution, reporting honest double-H lifetimes plus a
`lifetime_independent_alone_ns` counterfactual in provenance. Verified live
on path/star at R=1/64 (deterministic ×2, values match across picks).

Harness (`policy_sched.json`, new `both_search` rule): the shipped selector
at 70% error / 122 ms mean regret — it only pays off at high reuse
(path R > ~21, star R > ~540), since both searches are always charged.
Two-layer contract: gate invocation on a reuse threshold from history
(R-threshold rule: 0 errors), then select exactly. Pre-search skipping
needs a savings model and stays out of scope.

Robustness notes: calibration with repeats=2 produced 50× score swings vs
repeats=5 (NNLS on noisy sub-ms samples) — min-S picks survived, lifetime
magnitudes did not; first-call pyzx import lands in H_inc (runners warm it
up via rep -1). Runners keep repeats=5.

## RQ3 structural replay validation (2026-09-24, `trajectory.json`)
Phase probe first: downstream planning is `compile_recipes` branch assembly
(path 139/162 ms, star 33/83 ms); cone analysis 0.1 ms, cut enumeration
~0 ms. `cx_branches` costs 0.26 µs (Qiskit import ~300 ms hits once per
process — primed in runners, not a code issue).

Mechanism (`plan_batch._structural_hints`, experimental): template from
`explain()["search"]["structural_template"]` (topology key + partitions +
cuts); replay constrains compiles and skips beam; mismatch falls back with
`structural_miss` recorded (misses are measurable — required for hit rate).
8 bindings × path/star, oracle-checked: hit rate 1.0, decision fidelity
true (hinted tasks/scores == full per binding), errors ~1e-16, planning
29.8 → 4.8 ms (path), 107 → 4.3 ms (star). Guard test pins replay +
mismatch fallback in the suite.

## Compute discipline: success cases guard the spend (2026-09-24)

Power analysis (bootstrap over recorded repeats, `B3_cached` vs `B5_cached`
warm totals): 1.6 ms edges (path) resolve at n=3; 0.2 ms moderate-variance
edges (local) need n≈15–20; 0.2 ms high-variance edges (star) still P≈0.77
at n=30 — practically unresolvable, verdict "parity" instead of more
repeats. Pilot-5-then-extend rule; extrapolation past n=7 is indicative.
Single-shot `_one` timings drift ~30% across minutes on this host: only
interleaved medians count.

Disjoint fast path (`batch_planner.py`): when distinct identical-cone
groups are pairwise wire-disjoint in every variant, the joint beam can add
only whole-program unions, evaluated zero-cut-only (union-needing-cuts
skipped as flagged heuristic). disjoint_batch64: same 8-task winner, B5
cold 35.4 → 18.6 ms; fires nowhere else (path/star/overlap/local abstain,
verified). New unit test pins firing + winner preservation. R=1 keeps its
deterministic cap (≤12 states) rather than a hard skip: a one-shot beam win
is possible in principle, and the cap already bounds the spend.

Calibration mix (`extra_mix`, opt-in, default unchanged): H/RZ family
shifts fits ~10% but host drift across minutes is ~30% — inconclusive by
design. Drift-averaged medians (67 width-2 tasks × 7 shuffled repeats):
bias 1.029 default vs 1.044 mix. No bias to fix; no default change. The
earlier 1.24× readings were single-shot noise. Lesson recorded: substantive
calibration claims require runner-grade interleaved evidence.

## ZX trajectory + zero stress (`trajectory_zx.json`, 2026-09-24)

ZX path, 8 random ±1 rad bindings/case: hit rate 1.0, fidelity true,
errors ~1e-16, planning path 35→9.9 ms / star 250→8.4 ms. Rewrite is
structurally stable across VQE-scale perturbations here — planning reuse
survives arbitrary angles where LarQucut execution reuse is excluded.
All-zero binding: topology diverges → miss=1, fallback correct
(1 task, err 0.0), matching full planning. Both RQ3 halves validated.

## Tensor two-sided boundary (2026-09-24)

`tensor_compare/` (run_p4_p5 harness, B4/B5/B3 × 7): cold one-shot tensor
wins big on 4q fragments (B4/B5 0.90× path, 0.11× star); split shows
contraction 0.3–0.5 ms vs path search 28–33 ms re-run per call. Vs
uncached B4, joint wins from R≈2/10; vs path-cached tensor it loses
throughout at these sizes. Connected-12q probe inverts it: B4 prep 14.2 s
vs dense 3.0 ms (~4700×) — preparation, not contraction (26 ms), loses.
Batch planning wins where path search is expensive but dense evolution is
cheap, and vice versa. Recorded in PRIOR_ART_BOUNDARIES.md §2.

## Paper integration plan (2026-09-24, not yet written)

- §02 background: Quff/LarQucut-read-through/OBP+cutting/Huang-lightcones/
  qiskit-multi-obs-docs/Stollenwerk-ZX-PQC/Gragera-geometry paragraphs +
  references.bib entries (Huang 2026, Pal 2025, Stollenwerk 2023, Gragera).
- §04 methods: deterministic budget cap, calibrated expansion/scheduling
  costs, lifetime selector two-layer contract, disjoint guard, hint replay.
- §05 evaluation: policy selection-error tables (family C + held-out),
  tensor two-sided boundary, trajectory (none + zx) hit/fidelity/speedup,
  power-analysis repeat policy.
- §06 limits: noise-limited flips, calibration variance (repeats=2 vs 5),
  single-host scope, union-with-cuts heuristic risk.
- Abstract + conclusion refresh; rebuild PDF.
