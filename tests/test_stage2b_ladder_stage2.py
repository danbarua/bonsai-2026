"""Tier 1 checks on the ladder stage-2 driver, run locally with no network,
no GPU and no Colab session.

The driver itself cannot be exercised here -- it needs an A100, a bucket
and a real corpus. What CAN be pinned locally is everything that would
otherwise only be discovered by spending money: that the constants it
agrees on with the Makefile actually agree, that its module scope stays
importable without the repo's cloud dependencies, that the naming/reuse
decisions written into its docstrings are the naming and reuse the code
actually performs, and that every call site -- including the CNN closure,
genuinely new at this rung -- still binds against the real signatures.

Each of these has a specific failure it exists to prevent, named at the
test. None of them is a restatement of the driver's own source.
"""
import ast
import importlib
import inspect
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE2B_DIR = REPO_ROOT / "experiments" / "stage2b_denoising"
DRIVER_PATH = STAGE2B_DIR / "run_ladder_stage2.py"

sys.path.insert(0, str(STAGE2B_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _makefile import make_var as _make_var, recipes as _recipes  # noqa: E402

_ANY = object()
TARGET = "stage2b-ladder-stage2"


def _expanded(body, passes=3):
    for _ in range(passes):
        def substitute(match):
            value = _make_var(match.group(1))
            return value if value is not None else match.group(0)
        expanded = re.sub(r"\$\(([A-Za-z_][A-Za-z0-9_]*)\)", substitute, body)
        if expanded == body:
            break
        body = expanded
    return body


@pytest.fixture(scope="module")
def driver():
    """Import the driver with BONSAI_COMMIT guaranteed absent -- see
    `test_stage2b_ladder_stage1.py`'s identical fixture for why."""
    previous = os.environ.pop("BONSAI_COMMIT", None)
    try:
        module = importlib.import_module("run_ladder_stage2")
        yield module
    finally:
        if previous is not None:
            os.environ["BONSAI_COMMIT"] = previous


# ---- module scope stays cheap ----

def test_module_scope_imports_without_the_cloud_dependencies():
    """No module-level import may name anything outside the standard
    library, numpy, and `run_ladder_stage1` -- which is included here
    specifically because it is ITSELF stdlib+numpy-only at module scope
    (its own test pins that), so importing its staging constants does not
    reintroduce a cloud dependency. A hoisted `import jax` would make this
    file uncollectable."""
    tree = ast.parse(DRIVER_PATH.read_text(), filename=str(DRIVER_PATH))
    allowed = {"contextlib", "hashlib", "json", "os", "subprocess", "sys",
               "threading", "time", "traceback", "types", "numpy",
               "run_ladder_stage1"}
    offenders = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            offenders += [a.name for a in node.names if a.name.split(".")[0] not in allowed]
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in allowed:
                offenders.append(node.module)
    assert not offenders, (
        f"module-level imports outside stdlib+numpy+run_ladder_stage1: {offenders}. "
        f"Every repo import belongs inside load_modules().")


def test_importing_the_driver_does_not_run_it(driver):
    assert driver.OK_SENTINEL == "STAGE2_OK"
    assert driver.FAIL_SENTINEL == "STAGE2_FAIL"


def test_the_entry_guard_fires_on_bonsai_commit_not_only_on_main():
    source = DRIVER_PATH.read_text()
    assert '__name__ == "__main__" or os.environ.get(ENV_COMMIT)' in source


# ---- constants that two files have to agree on ----

def test_evolution_chunk_divides_the_rung_exactly(driver):
    import stage2b_partition as partition
    assert partition.STAGE2_SUBSET_SIZE % driver.EVOLVE_CHUNK == 0


def test_feature_dimension_follows_from_the_active_support(driver):
    assert driver.EXPECTED_FEATURE_DIM == 2 * driver.EXPECTED_N_ACTIVE - 2 == 1008


def test_the_smoke_banner_matches_stage_1s_locked_wording(driver):
    """The two drivers must agree on this string -- it is what makes stage
    2's stats-smoke artifact (when run) directly comparable to stage 1's."""
    import run_ladder_stage1
    assert driver.SMOKE_BANNER == run_ladder_stage1.SMOKE_BANNER == (
        "SMOKE OF THE MACHINERY ONLY -- IN-SAMPLE, TRAINING-SIDE, "
        "NON-INFERENTIAL, NOT A RESULT")


# ---- the reuse decisions: KMNIST staging and topologies ----

def test_kmnist_staging_is_imported_from_stage_1_not_redefined(driver):
    """"KMNIST inputs already staged -- reuse, don't re-stage" enforced as
    object identity, not merely equal values: a redefined dict could drift
    from stage 1's without any test catching it until a download 404s on
    the runtime."""
    import run_ladder_stage1
    assert driver.KMNIST_FILES is run_ladder_stage1.KMNIST_FILES
    assert driver.KMNIST_EXT == run_ladder_stage1.KMNIST_EXT
    assert driver.KMNIST_SUBDIR == run_ladder_stage1.KMNIST_SUBDIR


def test_kmnist_staging_stage_is_stage_1s_own_constant(driver):
    """`KMNIST_STAGING_STAGE` must equal stage 1's `LADDER_STAGE` -- via
    the import, not a hardcoded literal `1` that could silently stop
    matching if stage 1's own numbering ever changed."""
    import run_ladder_stage1
    assert driver.KMNIST_STAGING_STAGE == run_ladder_stage1.LADDER_STAGE == 1
    assert driver.LADDER_STAGE == 2   # stage 2's own, unaffected by the import


def test_topologies_are_reused_from_stage_1s_object_path_not_rebuilt(driver):
    """Topologies do not depend on which images are processed -- the same
    construction at every ladder stage -- so stage 2 downloads stage 1's
    already-verified artifact rather than re-running `build_all_topologies`
    under a second, potentially-diverging object name."""
    mods = driver.load_modules(str(REPO_ROOT))
    import stage2b_gcs as gcs
    name = gcs.object_path(stage=driver.KMNIST_STAGING_STAGE, condition=None,
                           kind="topologies", ext="npz", split="train")
    assert name == "stage2b/train/stage1/common/topologies.npz"
    source = DRIVER_PATH.read_text()
    # The specific call site, not a bare substring: `KMNIST_STAGING_STAGE`
    # also appears in `stage_kmnist`'s own (correct, unrelated) download
    # call, so a loose whole-file search would keep passing even if THIS
    # call site regressed to the driver's own LADDER_STAGE.
    assert ('_obj(mods, "topologies", "npz", stage=KMNIST_STAGING_STAGE)'
            in source), (
        "step1b_topologies must fetch the object under KMNIST_STAGING_STAGE "
        "(stage 1's own path), not driver's own LADDER_STAGE")


def test_encode_diagnostic_artifact_name_self_invalidates_on_step_count(driver):
    """Same self-invalidation discipline as stage 1's encoder-gate artifact
    (the amendment's own precedent): a future ENCODER_STEPS change mints a
    new object rather than silently resuming a stale one."""
    source = DRIVER_PATH.read_text()
    assert 'kind = f"encode_diagnostic_s{steps_now}"' in source, (
        "the encode-diagnostic object kind must be built from the LIVE "
        "ENCODER_STEPS value read at call time, not a hardcoded string")
    mods = driver.load_modules(str(REPO_ROOT))
    import stage2b_gcs as gcs
    name_now = gcs.object_path(stage=2, condition=None,
                               kind=f"encode_diagnostic_s{mods.encoder_gate.ENCODER_STEPS}",
                               ext="npz", split="train")
    name_at_old_default = gcs.object_path(stage=2, condition=None,
                                          kind="encode_diagnostic_s150", ext="npz",
                                          split="train")
    assert name_now != name_at_old_default


def test_local_paths_are_distinct_for_distinct_objects(driver):
    import stage2b_gcs as gcs
    names = [gcs.object_path(stage=2, condition=cond, kind=kind, ext=ext, split="train")
             for cond, kind, ext in (
                 (None, "corpus", "npz"), (None, "corruption", "npz"),
                 (None, "corruption_diagnostics", "npz"),
                 (None, "encode_diagnostic_s1200", "npz"),
                 (None, "ridge_cv", "json"), (None, "ridge_final", "npz"),
                 (None, "validation_corruption", "npz"), (None, "cnn_development", "npz"),
                 (None, "stats_smoke", "json"), (None, "stats_smoke", "txt"),
                 (None, "stage2_report", "json"), (None, "stage2_report", "txt"),
                 ("evolved_T", "theta_T", "npz"), ("evolved_T", "features", "npz"),
                 ("evolved_lattice", "theta_T", "npz"),
                 ("pre_evolution", "features", "npz"))
             ] + [gcs.object_path(stage=1, condition=None, kind=kind, ext="npz", split="train")
                  for kind in ("topologies", "corruption")]
    paths = [driver.local_path_for(n) for n in names]
    assert len(set(paths)) == len(paths), "two objects share a local path"


# ---- the Makefile has to supply what the driver reads ----

def test_the_make_target_exists_with_the_right_shape():
    assert TARGET in _recipes(), f"{TARGET} is not defined in the Makefile"


def test_the_make_target_sets_every_environment_variable_the_driver_reads(driver):
    required = {value for name, value in vars(driver).items()
                if name.startswith("ENV_") and isinstance(value, str)}
    assert required, "no ENV_* constants found -- this check has gone vacuous"
    body = _expanded(_recipes()[TARGET])
    missing = sorted(var for var in required if var not in body)
    print(f"\n[ladder2] driver reads {sorted(required)}")
    assert not missing, f"the recipe never sets {missing}"


def test_the_make_target_refuses_before_it_provisions():
    body = _recipes()[TARGET]
    first_refusal = min(body.index("REFUSING"), body.rindex("REFUSING"))
    assert first_refusal < body.index(") new -s"), (
        "a pre-flight refusal appears after session provisioning; it would fire "
        "only once a GPU is already billing")


def test_the_make_target_installs_equinox_and_optax():
    """The one dependency delta from stage 1: the CNN needs equinox/optax,
    which stage 1's own target correctly does not install."""
    body = _recipes()[TARGET]
    reinstall_lines = [l for l in body.splitlines() if ") reinstall " in l]
    assert len(reinstall_lines) == 1
    assert "equinox" in reinstall_lines[0] and "optax" in reinstall_lines[0], (
        f"CNN development needs equinox+optax installed on the runtime: "
        f"{reinstall_lines[0]}")


def test_stage_1s_own_target_does_not_gain_the_cnn_dependencies():
    """The converse of the above -- stage 1 has no CNN and must not pay for
    installing packages it never imports."""
    body = _recipes()["stage2b-ladder-stage1"]
    reinstall_lines = [l for l in body.splitlines() if ") reinstall " in l]
    assert len(reinstall_lines) == 1
    assert "equinox" not in reinstall_lines[0] and "optax" not in reinstall_lines[0]


def test_the_driver_is_uploaded_by_name_nowhere(driver):
    body = _recipes()[TARGET]
    uploads = [line for line in body.splitlines() if ") upload " in line]
    assert len(uploads) == 1, f"expected only the credentials upload, got {uploads}"
    assert "KEY_PATH" in uploads[0], f"unexpected upload: {uploads[0]}"


def test_stage_1_and_stage_2_use_distinct_sessions():
    """Stage 2's unconditional teardown must not be able to kill a session
    stage 1 still expects to be running, and vice versa."""
    session1 = _make_var("SESSION_2B_LADDER")
    session2 = _make_var("SESSION_2B_LADDER2")
    assert session1 is not None and session2 is not None
    assert session1 != session2


# ---- serialization helpers ----

def test_json_default_handles_the_numpy_types_the_record_carries(driver):
    import json
    import numpy as np
    payload = {"arr": np.arange(3), "f": np.float64(1.5), "i": np.int64(7),
               "b": np.bool_(True), "t": (1, 2)}
    loaded = json.loads(json.dumps(payload, default=driver._json_default))
    assert loaded == {"arr": [0, 1, 2], "f": 1.5, "i": 7, "b": True, "t": [1, 2]}


# ---- the call sites match the modules they call ----

def test_the_drivers_dependency_closure_resolves(driver):
    mods = driver.load_modules(str(REPO_ROOT))
    for name in ("ridge", "cnn", "corruption", "encoder_gate", "gcs", "partition",
                 "stats", "core", "topologies", "conditions", "verify_gpu"):
        assert getattr(mods, name) is not None


def test_every_call_the_driver_makes_binds_to_the_real_signature(driver):
    """Same discipline as stage 1's identical test -- extended to the CNN
    closure, genuinely new at this rung."""
    mods = driver.load_modules(str(REPO_ROOT))
    calls = [
        (mods.corruption.corrupt_corpus, (_ANY, "train", _ANY), {"alpha_bar": 0.5}),
        (mods.corruption.epsilon_for, ("train", 7), {}),
        (mods.corruption.forward_corrupt, (_ANY, _ANY, 0.5), {}),
        (mods.corruption.corruption_diagnostics, (_ANY, _ANY, _ANY, _ANY),
         {"labels": _ANY}),
        (mods.corruption.clip_rate_agreement, (_ANY, _ANY), {"alpha_bar": 0.5}),
        (mods.encoder_gate.encode_with_final_delta_batch, (_ANY, _ANY),
         {"seed": 0, "steps": 1200, "n_workers": 1}),
        (mods.topologies.build_all_topologies, (), {}),
        (mods.core.reference_node_features, (_ANY, 363), {}),
        (mods.core.evolve_on_graph, (_ANY, _ANY), {}),
        (mods.ridge.cross_validate_alpha, (_ANY, _ANY, _ANY), {}),
        (mods.ridge.ridge_equivalence_check, (_ANY, _ANY, _ANY), {}),
        (mods.ridge.fit_final, (_ANY, _ANY, 1.0), {}),
        (mods.ridge.ridge_predict, (_ANY, _ANY, 0), {}),
        (mods.ridge.clipped_per_image_mse, (_ANY, _ANY), {}),
        (mods.stats.run_stage2b_inference, (_ANY, _ANY), {"identity_key": "identity"}),
        (mods.gcs.ensure_artifact, (_ANY, _ANY),
         {"produce": _ANY, "bucket": _ANY, "force": True}),
        (mods.gcs.object_path, (),
         {"stage": 2, "condition": None, "kind": "corpus", "ext": "npz", "split": "train"}),
        (mods.gcs.download_file, (_ANY, _ANY), {"bucket": _ANY}),
        (mods.gcs.get_bucket, (), {"name": None, "credentials": _ANY}),
        (mods.gcs.checksum_backend, (), {}),
        (mods.conditions.path_segment, ("T",), {}),
        (mods.partition.Stage2BTrainingPartition, (_ANY,), {}),
        (mods.partition.Stage2BTrainingPartition.nested_development_subsets, (_ANY,),
         {"size": 5000, "prefix_size": 1000, "seed": 42, "stratified": True}),
        (mods.partition.Stage2BTrainingPartition.validation_labels, (_ANY,), {}),
        (mods.load_mnist, (_ANY,), {"gz": False}),
        (mods.local_converged_phases, (_ANY,), {"steps": 1200, "seed": 0}),
        # -- the CNN closure --
        (mods.cnn.build_active_support_mask, (_ANY,), {"expect_n_active": 505}),
        (mods.cnn.train_cnn_for_seed, (_ANY, _ANY, _ANY, _ANY, _ANY), {"seed": 0}),
        (mods.cnn.select_best_seed, (_ANY, _ANY), {}),
    ]
    failures = []
    for fn, args, kwargs in calls:
        try:
            inspect.signature(fn).bind(*args, **kwargs)
        except TypeError as exc:
            name = getattr(fn, "__qualname__", repr(fn))
            failures.append(f"{name}: {exc}")
    assert not failures, "the driver calls these in a way they no longer accept:\n" + \
        "\n".join(failures)


def test_the_batched_evolution_binding_is_the_batched_one(driver):
    source = DRIVER_PATH.read_text()
    assert "batched_evolve_on_graph_jax" in source
    assert "from evolve_on_graph_jax import batched_evolve_on_graph_jax" in source
    import evolve_on_graph_jax as ev
    assert ev.batched_evolve_on_graph_jax is not ev.evolve_on_graph_jax


def test_cnn_call_sites_use_train_cnn_for_seed_not_the_full_wrapper(driver):
    """Per-seed wall-clock is a named report item, and `train_best_of_seeds`
    does not expose it -- the driver must call `train_cnn_for_seed` +
    `select_best_seed` directly (the same public primitives the wrapper
    itself composes) rather than reimplementing seed selection from
    scratch, and rather than calling the wrapper and losing the timing."""
    source = DRIVER_PATH.read_text()
    assert "mods.cnn.train_cnn_for_seed(" in source
    assert "mods.cnn.select_best_seed(" in source


# ---- stats-smoke budget decision ----

def test_stats_smoke_budget_projection_is_derived_from_stage_1s_measurement(driver):
    """The projection basis is a real, cited number from stage 1's own
    report, not an invented constant."""
    assert driver.STATS_SMOKE_STAGE1_REFERENCE_N == 1000
    assert driver.STATS_SMOKE_STAGE1_REFERENCE_S > 0
    projected = (driver.STATS_SMOKE_STAGE1_REFERENCE_S
                * (5000 / driver.STATS_SMOKE_STAGE1_REFERENCE_N))
    assert projected < driver.STATS_SMOKE_BUDGET_S, (
        f"at n=5000 the projected stats-smoke cost ({projected:.1f}s) is expected to "
        f"stay under budget ({driver.STATS_SMOKE_BUDGET_S}s) -- if this now fails, the "
        f"driver's own live decision will legitimately skip the step, which is correct "
        f"behavior, but the expectation recorded here should be updated to match")


def test_driver_compiles_under_the_projects_interpreter():
    result = subprocess.run([sys.executable, "-m", "py_compile", str(DRIVER_PATH)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
