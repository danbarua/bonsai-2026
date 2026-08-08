"""An undeclared dependency can hide inside a skip.

`pytest.importorskip("X")` is the right construct for a capability this
machine may not have. It is the wrong construct for a package the project
depends on — there, it converts a broken environment into a quiet skip,
and a skip is not a failure. The suite stays green, the count moves by
one, and nobody diffs counts.

**The incident.** `test_crc32c_agrees_with_google_crc32c_where_it_is_installed`
checks that `stage2b_gcs.py`'s pure-Python CRC32C computes the same digest
as the library GCS actually uses. `google_crc32c` was never declared in
`pyproject.toml`; it arrived transitively through `google-cloud-storage`.
A bare `uv sync` pruned it on its own, the cross-check became a skip, and
`make stage2b-test` reported 825 passed where it had reported 826.

Nothing failed. `stage2b_gcs.py` falls back to pure Python by design, so
the fallback kept working — it just stopped being checked against the
thing it exists to match.

**What is derived here.** The set of `importorskip` calls comes from the
AST of every test file, not from a list. The declared set comes from
`pyproject.toml`. Neither is hand-maintained, so a new `importorskip`
added next month is checked on the day it lands rather than whenever
someone remembers this file exists.
"""
from __future__ import annotations

import ast
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"

# Where a first-party module may live. An `importorskip` naming one of
# these is legitimate and unrelated to packaging: those modules skip when
# an OPTIONAL import inside them fails, which is a capability question.
FIRST_PARTY_ROOTS = (
    REPO_ROOT / "experiments",
    REPO_ROOT / "src",
    REPO_ROOT / "tools",
)


def _normalise(name: str) -> str:
    """PEP 503-ish: `google_crc32c` and `google-crc32c` are one name."""
    return name.lower().replace("_", "-")


def declared_distributions(pyproject: Path | None = None) -> set[str]:
    """Every distribution the project declares, across all groups."""
    data = tomllib.loads((pyproject or REPO_ROOT / "pyproject.toml").read_text())
    specs = list(data.get("project", {}).get("dependencies", []))
    for group in data.get("dependency-groups", {}).values():
        specs.extend(s for s in group if isinstance(s, str))
    names = set()
    for spec in specs:
        # "google-crc32c>=1.5.0" / "pandas-stubs~=3.0.5" / "numpy"
        head = spec.split(";")[0].strip()
        for sep in (">=", "<=", "==", "~=", "!=", ">", "<", "["):
            head = head.split(sep)[0]
        names.add(_normalise(head.strip()))
    return names


def importorskip_modules(root: Path | None = None) -> dict[str, list[str]]:
    """{module name: [files that importorskip it]}, from the AST.

    From the AST rather than a grep, for the reason VACUOUS_TESTS #17 and
    #20 both record: a source search reads the spelling of the code and a
    comment satisfies it. Only a literal string argument is collected —
    a computed module name is not something this check can resolve, and
    silently ignoring one would be the under-coverage this file exists to
    prevent, so it is reported separately below.
    """
    found: dict[str, list[str]] = {}
    for path in sorted((root or TESTS_DIR).rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (func.attr if isinstance(func, ast.Attribute)
                    else func.id if isinstance(func, ast.Name) else None)
            if name != "importorskip" or not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                found.setdefault(first.value, []).append(path.name)
    return found


def first_party_modules() -> set[str]:
    """Module names importable from this repository's own source trees.

    Tests add the relevant `experiments/` directory to `sys.path` and then
    `importorskip` a module by bare name, so the check is on basenames.
    Package directories count too -- a `bonsai/` with an `__init__.py` is
    importable as `bonsai`.
    """
    names = set()
    for root in FIRST_PARTY_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            names.add(path.stem)
            if path.name == "__init__.py":
                names.add(path.parent.name)
    return names


def is_first_party(module: str) -> bool:
    return module.split(".")[0] in first_party_modules()


def test_every_importorskip_names_a_declared_or_first_party_module():
    """The guard. A third-party module skipped-on-import must be declared.

    Undeclared means nothing in the project asserts it should be present,
    so its absence is indistinguishable from a capability this machine
    legitimately lacks — which is exactly how a real dependency spent
    weeks being satisfied transitively and then vanished without a red
    test anywhere.
    """
    declared = declared_distributions()
    offenders = {}
    for module, files in importorskip_modules().items():
        if is_first_party(module):
            continue
        if _normalise(module) in declared:
            continue
        offenders[module] = files
    assert not offenders, (
        f"importorskip on undeclared third-party module(s): {offenders}. "
        f"An undeclared dependency wearing a skip cannot be told apart "
        f"from a capability this machine does not have -- declare it in "
        f"pyproject.toml so it lives and dies with its group, or make the "
        f"import hard so its absence is a failure")


def test_the_scan_actually_finds_importorskip_calls():
    """Anti-vacuity, and the lesson is recent enough to name.

    The guard above passes trivially over an empty set. If the AST walk
    breaks, or every `importorskip` is renamed or removed, it goes green
    while checking nothing -- the shape VACUOUS_TESTS #19 and #20 are
    both instances of. So the scan asserts it found the calls that exist.
    """
    modules = importorskip_modules()
    assert len(modules) >= 3, (
        f"the AST scan found only {sorted(modules)}; this suite has more "
        f"importorskip call sites than that, so the walk is broken and "
        f"the guard above is passing over an empty set")
    assert "google_crc32c" in modules, (
        "the module whose silent pruning prompted this file is no longer "
        "found by the scan -- either the test was removed, or the walk is")


def test_the_computed_module_case_is_absent_rather_than_ignored():
    """`importorskip(some_variable)` cannot be resolved by this check.

    None exist today. If one appears, this fails rather than letting the
    guard above quietly cover less than it appears to -- the difference
    between a check that does not apply and a check that silently skipped
    something is the whole subject of this file.
    """
    unresolvable = []
    for path in sorted(TESTS_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (func.attr if isinstance(func, ast.Attribute)
                    else func.id if isinstance(func, ast.Name) else None)
            if name != "importorskip" or not node.args:
                continue
            if not isinstance(node.args[0], ast.Constant):
                unresolvable.append(f"{path.name}:{node.lineno}")
    assert not unresolvable, (
        f"importorskip with a non-literal module name at {unresolvable}; "
        f"this check cannot resolve it, and an unresolvable call site is "
        f"a hole in the guard rather than an exemption from it")
