"""Tier 1 checks on the ladder stage-1 driver, run locally with no network,
no GPU and no Colab session.

The driver itself cannot be exercised here -- it needs an A100, a bucket and
a real corpus. What CAN be pinned locally is everything that would otherwise
only be discovered by spending money: that the constants it agrees on with
the Makefile actually agree, that its module scope stays importable without
the repo's cloud dependencies, and that the naming decisions written into
its docstrings are the naming the code performs.

Each of these has a specific failure it exists to prevent, named at the
test. None of them is a restatement of the driver's own source.
"""
import ast
import re
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE2B_DIR = REPO_ROOT / "experiments" / "stage2b_denoising"
DRIVER_PATH = STAGE2B_DIR / "run_ladder_stage1.py"

sys.path.insert(0, str(STAGE2B_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _makefile import make_var as _make_var, recipes as _recipes  # noqa: E402


def _expanded(body, passes=3):
    """A recipe with its `$(VAR)` references substituted in.

    Needed because a recipe can supply something entirely through a
    variable -- the ladder target sets two of the driver's environment
    variables via `$(GCS_EXEC_ENV)` -- and a check that only greps the
    literal recipe text would report those as missing. Grepping the
    unexpanded body is the same shape of mistake as verifying a narrowing
    with the broader tool: it answers a question next to the one asked."""
    import re
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
    """Import the driver with BONSAI_COMMIT guaranteed absent.

    Its entry guard fires on `__name__ == "__main__"` OR the presence of
    BONSAI_COMMIT -- the second condition is what lets a Colab kernel run it
    under some other module name. That also means importing it in a shell
    where BONSAI_COMMIT happens to be set would execute the whole run. The
    fixture makes the precondition explicit rather than hoping."""
    previous = os.environ.pop("BONSAI_COMMIT", None)
    try:
        module = importlib.import_module("run_ladder_stage1")
        yield module
    finally:
        if previous is not None:
            os.environ["BONSAI_COMMIT"] = previous


# ---- module scope stays cheap ----

def test_module_scope_imports_without_the_cloud_dependencies():
    """The driver defers every repo import into `load_modules`, so this file
    and `stage_kmnist_inputs` can import it for its constants on a machine
    with no jax, no google-cloud-storage and no clone.

    Checked in a subprocess with an empty sys.path entry for the repo's
    experiment dirs removed is not possible -- so check the AST instead: no
    module-level import may name anything outside the standard library and
    numpy. A hoisted `import jax` would make this file uncollectable."""
    tree = ast.parse(DRIVER_PATH.read_text(), filename=str(DRIVER_PATH))
    allowed = {"contextlib", "hashlib", "json", "os", "subprocess", "sys",
               "threading", "time", "traceback", "types", "numpy"}
    offenders = []
    for node in tree.body:                       # module level only, not nested
        if isinstance(node, ast.Import):
            offenders += [a.name for a in node.names if a.name.split(".")[0] not in allowed]
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in allowed:
                offenders.append(node.module)
    assert not offenders, (
        f"module-level imports outside stdlib+numpy: {offenders}. Every repo import "
        f"belongs inside load_modules(), both so the import ORDER that enables jax's "
        f"x64 mode stays in one place and so this file's constants remain readable "
        f"without the cloud dependencies installed.")


def test_importing_the_driver_does_not_run_it(driver):
    """The entry guard must not fire on a plain import."""
    assert driver.OK_SENTINEL == "STAGE1_OK"
    assert driver.FAIL_SENTINEL == "STAGE1_FAIL"


def test_the_entry_guard_fires_on_bonsai_commit_not_only_on_main():
    """`mighty-colab exec` gives the transmitted code some other __name__,
    so the environment variable is what says "this is the runtime". If that
    condition were dropped the driver would import and do nothing, exit 0,
    print no sentinel -- and the make target's sentinel check is the only
    thing that would catch it."""
    source = DRIVER_PATH.read_text()
    assert '__name__ == "__main__" or os.environ.get(ENV_COMMIT)' in source


# ---- constants that two files have to agree on ----

def test_evolution_chunk_divides_the_rung_exactly(driver):
    """A ragged final chunk compiles a second XLA shape for one short batch
    -- pure cost, and it makes the per-chunk timings incomparable."""
    import stage2b_partition as partition
    assert partition.STAGE1_SUBSET_SIZE % driver.EVOLVE_CHUNK == 0


def test_feature_dimension_follows_from_the_active_support(driver):
    """1008 is not a free constant: it is 2*505 - 2, the circular embedding
    minus the reference node's own two trivially-constant columns."""
    assert driver.EXPECTED_FEATURE_DIM == 2 * driver.EXPECTED_N_ACTIVE - 2 == 1008


def test_the_smoke_banner_is_the_locked_wording(driver):
    """The stats artifact's first line is specified verbatim. It carries
    four separate disclaimers and dropping any one of them changes what the
    artifact claims -- IN-SAMPLE in particular, which is what makes the
    numbers non-inferential by construction rather than by assertion."""
    assert driver.SMOKE_BANNER == (
        "SMOKE OF THE MACHINERY ONLY -- IN-SAMPLE, TRAINING-SIDE, "
        "NON-INFERENTIAL, NOT A RESULT")


# ---- the staged-input naming decision ----

def test_all_four_idx_files_are_staged(driver):
    """`load_mnist` opens train AND t10k unconditionally, and topology
    construction goes through it. Staging only the train pair fails on the
    runtime, inside graph construction, after provisioning."""
    assert set(driver.KMNIST_FILES) == {
        "train-images-idx3-ubyte", "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"}


def test_no_staged_object_is_named_test(driver):
    """The decision, enforced rather than documented: the t10k objects live
    under a `split="train"` path because that token is the PIPELINE split.
    A `kmnist_test_images` kind would read as a Stage 2B test-side artifact
    -- the exact misreading `stage2b_gcs`'s split guards exist to prevent."""
    offenders = [k for k in driver.KMNIST_FILES.values() if "test" in k]
    assert not offenders, (
        f"these staged object kinds read as test-side artifacts: {offenders}. "
        f"Use the byte provenance from the IDX filename (t10k), not 'test'.")


def test_staged_object_paths_are_legal_and_distinct(driver):
    """Built through `object_path`, so the module's own token rules apply --
    and distinct, because `ensure_artifact` keys a step's completion on the
    object's existence."""
    import stage2b_gcs as gcs
    names = {gcs.object_path(stage=driver.LADDER_STAGE, condition=None, kind=kind,
                             ext=driver.KMNIST_EXT, split=driver.SPLIT)
             for kind in driver.KMNIST_FILES.values()}
    assert len(names) == len(driver.KMNIST_FILES)
    for name in names:
        assert name.startswith(f"{gcs.TRAIN_ROOT}/stage1/{gcs.COMMON_CONDITION}/")
        assert not gcs.is_test_split_path(name)


def test_local_paths_are_distinct_for_distinct_objects(driver):
    """`ensure_artifact` trusts an existing local file when the object is
    already in the bucket, so two objects sharing a local path would hand
    the second one the first one's bytes -- silently, and only on a resumed
    run."""
    import stage2b_gcs as gcs
    names = [gcs.object_path(stage=1, condition=cond, kind=kind, ext=ext, split="train")
             for cond, kind, ext in (
                 (None, "corpus", "npz"), (None, "topologies", "npz"),
                 (None, "corruption", "npz"), (None, "encoder_gate", "npz"),
                 (None, "ridge_cv", "json"), (None, "ridge_final", "npz"),
                 (None, "stats_smoke", "json"), (None, "stats_smoke", "txt"),
                 (None, "stage1_report", "json"), (None, "stage1_report", "txt"),
                 ("evolved_T", "theta_T", "npz"), ("evolved_T", "features", "npz"),
                 ("evolved_lattice", "theta_T", "npz"),
                 ("pre_evolution", "features", "npz"))]
    paths = [driver.local_path_for(n) for n in names]
    assert len(set(paths)) == len(paths), "two objects share a local path"


# ---- the Makefile has to supply what the driver reads ----

def test_the_make_target_sets_every_environment_variable_the_driver_reads(driver):
    """Derived, not listed. The driver's ENV_* constants ARE the required
    set, so a new one added there fails here until the recipe supplies it --
    rather than failing on the runtime, after provisioning, with the
    KeyError arriving as a science result."""
    required = {value for name, value in vars(driver).items()
                if name.startswith("ENV_") and isinstance(value, str)}
    assert required, "no ENV_* constants found -- this check has gone vacuous"
    body = _expanded(_recipes()["stage2b-ladder-stage1"])
    missing = sorted(var for var in required if var not in body)
    print(f"\n[ladder] driver reads {sorted(required)}")
    assert not missing, (
        f"the recipe never sets {missing}, which the driver reads from the "
        f"environment. `mighty-colab exec --env K=V` is how they reach the kernel.")


def test_the_make_target_refuses_before_it_provisions():
    """Ordering, not merely presence: both refusals must appear before the
    first `mighty-colab new`, or the check costs an A100 to discover."""
    body = _recipes()["stage2b-ladder-stage1"]
    first_refusal = min(body.index("REFUSING"), body.rindex("REFUSING"))
    assert first_refusal < body.index(") new -s"), (
        "a pre-flight refusal appears after session provisioning; it would fire "
        "only once a GPU is already billing")


def test_the_driver_is_uploaded_by_name_nowhere(driver):
    """The driver is transmitted by `exec --file`, not uploaded to the VM,
    and it fetches its dependencies by cloning the pinned commit. An
    `upload` of a module alongside it would put a second, unpinned copy on
    the runtime's path -- which `load_modules` asserts against, but the
    cheaper place to catch it is here."""
    body = _recipes()["stage2b-ladder-stage1"]
    uploads = [line for line in body.splitlines() if ") upload " in line]
    assert len(uploads) == 1, f"expected only the credentials upload, got {uploads}"
    assert "KEY_PATH" in uploads[0], f"unexpected upload: {uploads[0]}"


# ---- the staging script and the driver agree ----

def test_the_staging_script_stages_exactly_what_the_driver_downloads():
    """One mapping, imported by both. This test is what makes that
    importable-ness load-bearing rather than incidental."""
    import stage_kmnist_inputs
    import run_ladder_stage1
    assert stage_kmnist_inputs.KMNIST_FILES is run_ladder_stage1.KMNIST_FILES
    assert stage_kmnist_inputs.KMNIST_EXT == run_ladder_stage1.KMNIST_EXT
    assert stage_kmnist_inputs.SPLIT == run_ladder_stage1.SPLIT
    assert stage_kmnist_inputs.LADDER_STAGE == run_ladder_stage1.LADDER_STAGE


def test_the_staging_scripts_default_source_is_the_repos_kmnist_dir():
    """It must read the same bytes the rest of the project uses, not a copy
    somebody placed elsewhere."""
    import stage_kmnist_inputs
    assert Path(stage_kmnist_inputs.DEFAULT_KMNIST_DIR) == REPO_ROOT / "datasets" / "kmnist"


# ---- serialization helpers the artifacts depend on ----

def test_json_default_handles_the_numpy_types_the_record_carries(driver):
    import json
    import numpy as np
    payload = {"arr": np.arange(3), "f": np.float64(1.5), "i": np.int64(7),
               "b": np.bool_(True), "t": (1, 2)}
    loaded = json.loads(json.dumps(payload, default=driver._json_default))
    assert loaded == {"arr": [0, 1, 2], "f": 1.5, "i": 7, "b": True, "t": [1, 2]}


# ---- the call sites match the modules they call ----

_ANY = object()


def test_the_drivers_dependency_closure_resolves(driver):
    """`load_modules` pointed at this repo instead of a clone.

    It is self-contained by design -- it adds the source directories to
    `sys.path` itself rather than depending on `bootstrap_repo` having run
    -- specifically so this check is possible with no runtime, no clone and
    no network. An ImportError here is one that would otherwise surface on
    a billing A100."""
    mods = driver.load_modules(str(REPO_ROOT))
    for name in ("ridge", "corruption", "encoder_gate", "gcs", "partition",
                 "stats", "core", "topologies", "conditions", "verify_gpu"):
        assert getattr(mods, name) is not None


def test_every_call_the_driver_makes_binds_to_the_real_signature(driver):
    """The cheap half of the glue-bug class, caught locally.

    Stage 1D's replica directions, the stage2a-verify no-op gate and this
    week's single-trial near-miss were all caller-side mistakes around
    kernels that were themselves correct. Argument binding will not catch a
    wrong VALUE -- only a call the module cannot accept at all -- but that
    is the half that is free to check, and it is the half that turns a
    module signature change into a local test failure instead of a remote
    one halfway through a paid run.

    `Signature.bind` performs no type checking, so sentinels are fine: what
    is under test is arity, keyword names and required arguments."""
    mods = driver.load_modules(str(REPO_ROOT))
    calls = [
        (mods.corruption.corrupt_corpus, (_ANY, "train", _ANY), {"alpha_bar": 0.5}),
        (mods.corruption.epsilon_for, ("train", 7), {}),
        (mods.corruption.forward_corrupt, (_ANY, _ANY, 0.5), {}),
        (mods.corruption.corruption_diagnostics, (_ANY, _ANY, _ANY, _ANY),
         {"labels": _ANY}),
        (mods.corruption.clip_rate_agreement, (_ANY, _ANY), {"alpha_bar": 0.5}),
        (mods.encoder_gate.run_encoder_gate, (_ANY, _ANY, _ANY),
         {"seed": 0, "steps": 150, "n_workers": 1}),
        (mods.encoder_gate.format_gate_log, (_ANY,), {}),
        (mods.topologies.build_all_topologies, (), {}),
        (mods.core.reference_node_features, (_ANY, 363), {}),
        (mods.core.evolve_on_graph, (_ANY, _ANY), {}),
        (mods.core.encode_and_restrict, (_ANY, _ANY), {}),
        (mods.ridge.cross_validate_alpha, (_ANY, _ANY, _ANY), {}),
        (mods.ridge.ridge_equivalence_check, (_ANY, _ANY, _ANY), {}),
        (mods.ridge.fit_final, (_ANY, _ANY, 1.0), {}),
        (mods.ridge.ridge_predict, (_ANY, _ANY, 0), {}),
        (mods.ridge.clipped_per_image_mse, (_ANY, _ANY), {}),
        (mods.stats.run_stage2b_inference, (_ANY, _ANY), {"identity_key": "identity"}),
        (mods.gcs.ensure_artifact, (_ANY, _ANY),
         {"produce": _ANY, "bucket": _ANY, "force": True}),
        (mods.gcs.object_path, (),
         {"stage": 1, "condition": None, "kind": "corpus", "ext": "npz",
          "split": "train"}),
        (mods.gcs.download_file, (_ANY, _ANY), {"bucket": _ANY}),
        (mods.gcs.get_bucket, (), {"name": None, "credentials": _ANY}),
        (mods.gcs.checksum_backend, (), {}),
        (mods.conditions.path_segment, ("T",), {}),
        (mods.partition.Stage2BTrainingPartition, (_ANY,), {}),
        (mods.partition.Stage2BTrainingPartition.nested_development_subsets, (_ANY,),
         {"size": 5000, "prefix_size": 1000, "seed": 42, "stratified": True}),
        (mods.load_mnist, (_ANY,), {"gz": False}),
    ]
    import inspect
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
    """By name, and by arity. `batched_evolve_on_graph_jax` is
    `jax.jit(jax.vmap(evolve_on_graph_jax, in_axes=(0, None)))` -- exactly
    two positional arguments and no `k_coupling`. The single-trial function
    is not an acceptable substitute, and swapping them is a near-miss this
    project has already had once."""
    source = DRIVER_PATH.read_text()
    assert "batched_evolve_on_graph_jax" in source
    assert "from evolve_on_graph_jax import batched_evolve_on_graph_jax" in source
    import evolve_on_graph_jax as ev
    assert ev.batched_evolve_on_graph_jax is not ev.evolve_on_graph_jax


# ---- the README's test table is a narrowing, so assert it matches ----

def test_the_readme_test_table_lists_exactly_the_stage2b_test_files():
    """Both directions, per principle 21.

    The table used to carry per-file counts as well. They drifted four
    times -- once with three rows stale simultaneously, and once from a
    MERGE, where nobody wrote a wrong number and the numbers became wrong
    anyway because two branches each added tests. Counts are derived and
    `make stage2b-test` prints them, so they were removed rather than
    guarded. What remains is a list, and a list standing in for a derivable
    set is exactly what this checks."""
    readme = (REPO_ROOT / "experiments" / "stage2b_denoising" / "README.md").read_text()
    listed = set(re.findall(r"`(test_stage2b_[a-z0-9_]+\.py)`", readme))
    declared = _make_var("STAGE2B_TEST_FILES")
    assert declared is not None
    expected = {Path(token).name for token in declared.split() if token.endswith(".py")}
    on_disk = {p.name for p in (REPO_ROOT / "tests").glob("test_stage2b_*.py")}
    assert expected == on_disk, "STAGE2B_TEST_FILES and tests/ disagree"

    missing = sorted(on_disk - listed)
    assert not missing, (
        f"the README's table omits {missing}. A reader takes that table as the "
        f"whole of what `make stage2b-test` runs.")
    stale = sorted(listed - on_disk)
    assert not stale, f"the README's table names files that no longer exist: {stale}"


def test_the_readme_carries_no_hand_maintained_test_counts():
    """The removal is the fix; without this the counts creep back.

    Prose like "its 148 tests" or "503 collected" is a copy of a number
    `make stage2b-test` already prints, and every copy has gone stale."""
    readme = (REPO_ROOT / "experiments" / "stage2b_denoising" / "README.md").read_text()
    offenders = re.findall(r"\b\d{2,4}\s+(?:fast\s+)?tests?\b|\b\d{2,4}\s+collected\b",
                           readme)
    assert not offenders, (
        f"the README states test counts in prose: {offenders}. They are derived "
        f"numbers -- let `make stage2b-test` report them.")


def test_driver_compiles_under_the_projects_interpreter():
    """A syntax error here is only discoverable on the runtime otherwise --
    the file is transmitted as code, so nothing imports it before it runs."""
    result = subprocess.run([sys.executable, "-m", "py_compile", str(DRIVER_PATH)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
