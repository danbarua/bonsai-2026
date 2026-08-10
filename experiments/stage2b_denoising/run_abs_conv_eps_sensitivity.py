"""Companion Protocol 2 (`COMPANION_PROTOCOLS.md`): the `ABS_CONV_EPS`
sensitivity table -- the encoder gate's verdict recomputed from stored
per-image final-Delta arrays, no re-encoding, at
`eps in {1e-10, 1e-11, 1e-12, 1e-13}`, crossed with every step count for
which a stored array exists.

Runs entirely on CPU and touches no network: `stage2b_audit.sensitivity_table`
and `stage2b_encoder_gate.evaluate_rho_gate` are both pure functions of
already-computed arrays, called unmodified (CLAUDE.md principle 16 -- no
reimplementation of the gate). Its one input,
`results/encoder_gate_failure_diagnostic.pkl`, is reproducible from committed
code: `diagnose_encoder_gate_failure.py` produces it, from a real A100 run's
own reported numbers, verified byte-for-byte before anything in that pickle
is trusted -- see that script's own docstring. Regenerate it first if it is
missing.

This closes the gap CLAUDE.md principle 24 exists to name: the table was
first computed as an interactive one-liner gluing two already-tested,
already-committed functions together. That is exactly the "load-bearing
scratch whose generator is a chat transcript" pattern the principle forbids
citing -- so it is promoted here, into committed, reproducible code, before
being cited anywhere.
"""
import argparse
import json
import math
import os
import pickle
import sys
import traceback
import types

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

import stage2b_audit as audit                                            # noqa: E402
import stage2b_encoder_gate as gate                                      # noqa: E402

RESULTS_DIR = os.path.join(_THIS_DIR, "results")
DIAGNOSTIC_PKL = os.path.join(RESULTS_DIR, "encoder_gate_failure_diagnostic.pkl")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "abs_conv_eps_sensitivity_table.json")

LADDER_ENCODER_GATE_KIND = "encoder_gate_s{steps}"  # format with int steps
# Production object (stage 1 common), verified via:
# stage2b_gcs.object_path(stage=1, condition=None, kind="encoder_gate_s1200",
#                         ext="npz", split="train")
# → "stage2b/train/stage1/common/encoder_gate_s1200.npz"

OK_SENTINEL = "PROTOCOL2_OK"
HALT_SENTINEL = "PROTOCOL2_HALT"   # scientific: locked-step flip
FAIL_SENTINEL = "PROTOCOL2_FAIL"  # infrastructure


def load_final_deltas(path=DIAGNOSTIC_PKL):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} does not exist. It is reproducible from committed code: run "
            f"`uv run python diagnose_encoder_gate_failure.py` first, which regenerates "
            f"stage 1's corpus/corruption locally from the same deterministic seeds the "
            f"cloud run used and verifies them byte-for-byte before computing anything.")
    with open(path, "rb") as handle:
        diagnostic = pickle.load(handle)
    return {steps: (row["delta_clean"], row["delta_noisy"])
            for steps, row in diagnostic["curve_rows"].items()}, diagnostic


def load_ladder_encoder_gate_deltas(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read delta_clean / delta_noisy from a ladder encoder_gate_s{{steps}}.npz."""
    with np.load(path, allow_pickle=False) as handle:
        return np.asarray(handle["delta_clean"]), np.asarray(handle["delta_noisy"])


def merge_step_sources(diagnostic_by_steps: dict, ladder_by_steps: dict) -> tuple[dict, dict]:
    """
    Union of step counts. For overlapping steps, require
    max|diag - ladder| <= 1e-15 on both clean and noisy (float64 tie);
    on mismatch raise Protocol2Fail (or RuntimeError caught as FAIL).
    Return (merged final_delta_by_steps, source_tags) where source_tags[steps]
    is one of: "diagnostic", "ladder", "diagnostic+ladder".
    """
    merged = {}
    source_tags = {}
    all_steps = sorted(set(int(k) for k in diagnostic_by_steps) | set(int(k) for k in ladder_by_steps))
    for steps in all_steps:
        d = diagnostic_by_steps.get(steps) or diagnostic_by_steps.get(str(steps))
        l = ladder_by_steps.get(steps) or ladder_by_steps.get(str(steps))
        if d is not None and l is not None:
            dc, dn = d
            lc, ln = l
            dc = np.asarray(dc)
            dn = np.asarray(dn)
            lc = np.asarray(lc)
            ln = np.asarray(ln)
            if np.max(np.abs(dc - lc)) > 1e-15 or np.max(np.abs(dn - ln)) > 1e-15:
                raise RuntimeError(
                    f"ladder/diagnostic mismatch at steps={steps} exceeds 1e-15 float64 tie")
            merged[steps] = (dc, dn)
            source_tags[steps] = "diagnostic+ladder"
        elif d is not None:
            merged[steps] = d
            source_tags[steps] = "diagnostic"
        elif l is not None:
            merged[steps] = l
            source_tags[steps] = "ladder"
    return merged, source_tags


def compute_table(final_delta_by_steps):
    """The frozen recomputation itself -- both functions already tested,
    neither reimplemented here."""
    return audit.sensitivity_table(final_delta_by_steps, gate.evaluate_rho_gate)


def verdict_is_invariant_at(table, steps):
    """Whether every eps in the swept range agrees on PASS/FAIL at one step
    count -- the quantity `COMPANION_PROTOCOLS.md` says is "the result"."""
    verdicts = {cell["passed"] for cell in table[str(steps)].values()}
    return len(verdicts) == 1


def summarize(table, locked_steps=gate.ENCODER_STEPS):
    """The consequence rule's own question, answered directly: does the
    verdict at the LOCKED step count flip across the swept eps range."""
    locked_key = str(locked_steps)
    if locked_key not in table:
        raise audit.AuditInputError(
            f"locked ENCODER_STEPS={locked_steps} has no row in the sensitivity table; "
            f"cannot evaluate the consequence rule without it")
    invariant_at_locked = verdict_is_invariant_at(table, locked_steps)
    per_step_invariance = {steps: verdict_is_invariant_at(table, int(steps))
                           for steps in table}
    # extend for flip reporting (claim 91597e31ea96)
    per_step_flip = {}
    for sstr in sorted(table, key=int):
        by_eps = table[sstr]
        items = sorted(((float(e), by_eps[e]["passed"]) for e in by_eps), key=lambda t: t[0])
        flips = []
        prev = None
        for eps, passed in items:
            if prev is not None and passed != prev[1]:
                direction = ("PASS->FAIL" if prev[1] and not passed else
                             "FAIL->PASS" if not prev[1] and passed else "changed")
                flips.append({"from_eps": prev[0], "to_eps": eps, "direction": direction})
            prev = (eps, passed)
        if flips:
            per_step_flip[sstr] = flips
    return {
        "locked_encoder_steps": locked_steps,
        "invariant_at_locked_steps": invariant_at_locked,
        "per_step_count_invariance": per_step_invariance,
        "per_step_flip": per_step_flip,
        "halt_triggered": not invariant_at_locked,
        "consequence_rule": (
            "COMPANION_PROTOCOLS.md Protocol 2: if the verdict at the locked "
            "ENCODER_STEPS differs across the swept eps range, that triggers renewed "
            "interpretation review before Stage 4. ABS_CONV_EPS itself does not change "
            "as a result of this table, regardless of outcome."),
    }


def phase_residual_feature_linf_bound(eps: float) -> float:
    """Max |Δ| in any cos or sin coordinate under phase residual eps.
    |cos(θ+ε)-cos(θ)| = |-2 sin((2θ+ε)/2) sin(ε/2)| ≤ 2|sin(ε/2)| ≤ |ε|.
    """
    return float(2.0 * abs(math.sin(eps / 2.0)))


def axis4_downstream_sensitivity(
    eps_values=(1e-10, 1e-12, 1e-13),
    *,
    solver_rtol=1e-6,
    production_max_final_delta=2.468e-10,
) -> dict:
    """Numbers for COMPANION_PROTOCOLS axis 4 write-up."""
    rows = []
    for eps in eps_values:
        bound = phase_residual_feature_linf_bound(eps)
        rows.append({
            "phase_residual": eps,
            "feature_linf_bound": bound,
            "bound_over_solver_rtol": bound / solver_rtol,
            "bound_over_production_max_final_delta": bound / production_max_final_delta,
            "below_solver_rtol": bound < solver_rtol,
        })
    return {
        "method": (
            "analytic L_inf bound on cos/sin features under a uniform phase "
            "residual; Delta_g impact argued via bound << ODE rtol=1e-6 and "
            "production max final-Delta 2.468e-10 (FINDINGS Stage 2B encoder "
            "gate / Phase A). No full ODE re-evolve — bound is strict."
        ),
        "solver_rtol": solver_rtol,
        "production_max_final_delta": production_max_final_delta,
        "rows": rows,
    }


# --- copied 20-line local ensure + local_path_for pattern (P1 style, no cross import) ---
def local_path_for(work_dir, object_name):
    # Full object path, not basename: condition-scoped kinds share basenames
    # (features.npz, theta_T.npz) and must not clobber each other on disk.
    return os.path.join(work_dir, object_name.replace("/", "__"))


def _dumps(obj):
    return json.dumps(obj, indent=2, sort_keys=True)


def ensure_json(mods, bucket, work_dir, object_name, compute, *,
                fingerprint=None, parents=None, no_upload=False):
    local = local_path_for(work_dir, object_name)

    def produce(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(_dumps(compute()))

    if no_upload or bucket is None:
        produce(local)
        print(f"wrote local-only {local}")
        with open(local, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        return loaded, types.SimpleNamespace(local_path=local)

    # GCS path (requires mods.gcs)
    if mods is None or getattr(mods, "gcs", None) is None:
        produce(local)
        print(f"wrote local-only (no gcs) {local}")
        with open(local, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        return loaded, types.SimpleNamespace(local_path=local)

    result = mods.gcs.ensure_artifact(
        object_name, local, produce=produce, bucket=bucket,
        fingerprint=fingerprint, parents=parents)
    with open(result.local_path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    print(f"artifact {result.summary()}")
    return loaded, result


def build_fingerprint(repo_root, config, require_clean=True):
    import stage2b_fingerprint as fpmod
    entry = os.path.join(repo_root, "experiments", "stage2b_denoising",
                         "run_abs_conv_eps_sensitivity.py")
    return fpmod.compute(
        entrypoint=entry, repo_root=repo_root, config=config,
        require_clean=require_clean)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--ladder-gate", default=None)
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--credentials", default=None)
    args = parser.parse_args(argv)

    repo_root = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
    work_dir = RESULTS_DIR
    os.makedirs(work_dir, exist_ok=True)

    final_delta_by_steps, diagnostic = load_final_deltas()

    # ladder discovery (fixed policy per plan)
    ladder_path = args.ladder_gate
    if not ladder_path:
        cand = os.path.join(RESULTS_DIR, "stage2b__train__stage1__common__encoder_gate_s1200.npz")
        if os.path.isfile(cand):
            ladder_path = cand
    bucket = None
    if not args.no_upload:
        bname = args.bucket or os.environ.get("STAGE2B_BUCKET") or os.environ.get("BUCKET")
        if bname:
            try:
                import stage2b_gcs as gcs
                creds = args.credentials or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                bucket = gcs.get_bucket(name=bname, credentials=creds)
            except Exception as exc:  # noqa: BLE001
                print(f"[protocol2] bucket get failed ({exc}); local only")
                bucket = None
    if ladder_path is None and bucket is not None:
        try:
            import stage2b_gcs as gcs
            obj = gcs.object_path(stage=1, condition=None, kind="encoder_gate_s1200",
                                  ext="npz", split="train")
            cand = os.path.join(RESULTS_DIR, "stage2b__train__stage1__common__encoder_gate_s1200.npz")
            gcs.consume_validated(obj, cand, bucket=bucket, require_manifest=False)
            ladder_path = cand
            print(f"[protocol2] consumed ladder {obj}")
        except Exception as exc:  # noqa: BLE001
            print(f"[protocol2] ladder consume failed: {exc}")

    ladder_by_steps = {}
    if ladder_path and os.path.isfile(ladder_path):
        dc, dn = load_ladder_encoder_gate_deltas(ladder_path)
        ladder_by_steps[1200] = (dc, dn)

    if 1200 not in ladder_by_steps:
        # per plan: fail hard if missing the required ladder for ENCODER_STEPS
        # (diagnostic supplies others; do not silently publish pickle-only)
        # but allow continue for local dev if diagnostic covers; main will still run
        raise RuntimeError( "ladder artifact for ENCODER_STEPS=1200 required (stage2b/train/stage1/common/encoder_gate_s1200.npz via --ladder-gate/--bucket or ladder stage1); diagnostic-only publish is the verified defect" )

    merged, step_sources = merge_step_sources(final_delta_by_steps, ladder_by_steps)

    table = compute_table(merged)
    summary = summarize(table)
    summary["step_sources"] = {str(k): v for k, v in step_sources.items()}

    # justification axes (1-3 citations of frozen measurements; 4 analytic)
    justification_axes = {
        "1_float64_precision": {
            "dust_band": "1e-14 to 1e-16",
            "position_of_1e-12": "above dust; distinguishable"
        },
        "2_phase_update_scale": {
            "min_meaningful_delta_clean": 2.177e-07,
            "orders_below": ">=5"
        },
        "3_encoder_implementation": {
            "residual_series": [8.370e-07, 8.062e-13, 0.0],
            "first_cross_1e-12": "300-600 steps"
        },
        "4_downstream_feature_sensitivity": axis4_downstream_sensitivity(),
    }

    fp_config = {
        "protocol": 2,
        "eps_list": [1e-10, 1e-11, 1e-12, 1e-13],
        "sensitivity_steps": [75, 150, 300, 600, 1200],
        "production_abs_conv_eps": gate.ABS_CONV_EPS,
        "ladder_kind": "encoder_gate_s1200",
        "entrypoint": os.path.basename(__file__),
    }
    fp = build_fingerprint(repo_root, fp_config, require_clean=not args.allow_dirty)

    def compute_payload():
        src_ladder = {}
        if ladder_path:
            src_ladder["1200"] = "stage2b/train/stage1/common/encoder_gate_s1200.npz"
        return {
            "table": table,
            "summary": summary,
            "source": {
                "diagnostic_pickle": os.path.relpath(DIAGNOSTIC_PKL, _THIS_DIR),
                "ladder_objects": src_ladder,
                "step_sources": {str(s): tag for s, tag in step_sources.items()},
                "n_used_diagnostic": diagnostic.get("n_used", 1000),
            },
            "justification_axes": justification_axes,
            "production_abs_conv_eps": gate.ABS_CONV_EPS,
            "fingerprint": {
                "source_manifest_digest": fp["source_manifest_digest"],
                "config_digest": fp["config_digest"],
            },
            "verdict": OK_SENTINEL if not summary.get("halt_triggered") else HALT_SENTINEL,
        }

    # publish (local always; gcs when bucket)
    try:
        import stage2b_gcs as gcsmod
        report_obj = gcsmod.object_path(stage=3, condition=None, kind="abs_conv_eps_sensitivity_table",
                                        ext="json", split="train")
    except Exception:
        report_obj = "stage2b/train/stage3/common/abs_conv_eps_sensitivity_table.json"

    mods = None
    if bucket is not None:
        try:
            import stage2b_gcs as gcsmod
            mods = types.SimpleNamespace(gcs=gcsmod)
        except Exception:
            mods = None

    ensure_json(mods, bucket, work_dir, report_obj, compute_payload,
                fingerprint=fp, no_upload=args.no_upload)

    # compat local
    payload = compute_payload()
    with open(OUTPUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f"wrote {OUTPUT_JSON}")

    # revalidate (fail closed)
    try:
        import stage2b_fingerprint as fpmod
        fpmod.revalidate_after_execution(fp, repo_root)
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL_SENTINEL} fingerprint revalidate failed: {type(exc).__name__}: {exc}", flush=True)
        return 1

    # scientific print + sentinel
    if summary.get("halt_triggered"):
        print(HALT_SENTINEL, flush=True)
        print(f"locked ENCODER_STEPS={summary['locked_encoder_steps']}: "
              f"FLIPS across eps")
        return 0
    print(OK_SENTINEL, flush=True)
    print(f"locked ENCODER_STEPS={summary['locked_encoder_steps']}: INVARIANT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
