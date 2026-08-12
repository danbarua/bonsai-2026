# Diagnosing a remote run from its log

A one-pager for the failure this project keeps paying for: a job dies on a
billing VM, and the agent driving it reconstructs the cause from the
repository instead of reading what the log already printed.

Every example here is from a single session on 2026-08-12, in which the same
job was launched four times.

## 1. Read the whole log first. It is the diagnosis, not a hint.

```bash
mighty-colab log -s <session> --tail | sed 's/\x1b\[[0-9;]*m//g'
```

The ANSI strip matters: the raw log is full of colour codes, and a
notification excerpt or a `grep` will show you a fragment that reads like a
bare error type with no context.

A real one, in full, at 2,326 bytes:

```
ModuleNotFoundError                       Traceback (most recent call last)
<mighty-colab-exec:run_gauge_comparison_2a_gpu.py> in <cell line: 0>()
     39 from sklearn.preprocessing import StandardScaler
     40
---> 41 import stage2a_core as s2a
     42 import stage2a_classifier_jax as clf_jax

/content/stage2a_core.py in <module>
     28 sys.path.insert(0, os.path.join(_THIS_DIR, "..", "stage1b2_structured_transformation"))
     29
---> 30 from build_stage1d_constructions import build_and_verify_T
ModuleNotFoundError: No module named 'build_stage1d_constructions'
```

That contains the whole answer — **what** failed, **where** in your file,
**where** in the dependency, and **why** (the `sys.path.insert` on line 28,
printed as context, resolves to a directory that exists only in the repo).

What happened instead: the import graph got grepped out of the repository to
reconstruct exactly this. Twenty minutes, and it produced a worse fix than
the log implied.

## 2. Know which frame is yours

`exec -f` prepends a prelude setting `sys.argv`, `__name__` and a synthetic
`__file__`, so the top frame names **your script**:

```
<mighty-colab-exec:run_gauge_comparison_2a_gpu.py> in <cell line: 0>()
```

Without it that frame is `<ipython-input-3-...>` and the line number points
at nothing. Frames below it are real files at real paths (`/content/...`)
with real line numbers. Read top-down: the top frame is the line *you* wrote,
each frame below is what it reached.

## 3. The log tells you the FIRST failure, not every failure

In the trace above, only line 30 failed. Lines 31 and 32 — two more imports,
one of them a package — were never reached. "Fix the module the error named"
buys one more provision and one more failure.

**Read what the failing line implies about its neighbours**, then fix the
class of problem, not the instance.

## 4. Distinguish "the job failed" from "the CLI failed"

Under `--json` the CLI exits 0 whenever it completed its transaction, even if
the remote job raised. The job's outcome is in the envelope:

```bash
mighty-colab --json log -s <session> --tail
```

| status | meaning |
|---|---|
| `running` | pid alive, no sidecar yet |
| `ok` | finished, did not raise |
| `job_raised` | finished, raised — with `exit_code` and `reason` |
| `error` + `worker_terminated` | pid gone, no sidecar: killed, not crashed |

That last row is the one grep cannot see. **An OOM kill or a backend
teardown prints no traceback**, so a loop that decides completion by
searching the log for a sentinel or `Traceback` waits until its timeout while
nothing is running. Poll the status; use the sentinel only as an additional
check that a clean exit actually reached its verdict.

Poll incrementally with `--since-offset <next_offset>` so following a long
job costs the log's length once, not once per poll.

## 5. Check the log is THIS run's

`--output-log` is a fixed path and is not truncated for you. A relaunch that
fails early leaves the previous run's bytes in place, and:

- a monitor tailing from byte zero replays an old traceback as if it were
  live (this cost a false diagnosis at 13:42 against a log written at 13:35);
- worse, a success check that greps for a sentinel matches the **previous**
  run's success and reports a job that wrote nothing as passing.

Truncate at submit (`: > "$LOG"`), or compare `stat -f %Sm` against now
before believing a line.

## 6. Gate on runtime output, never on recipe text

`make` echoes each recipe before running it, so the command's own source
appears in the captured output. A watcher gating on `grep "started as pid"`
matched the `echo "... started as pid ..."` **inside the recipe** and fired
before anything had been submitted.

Match a string only the CLI emits (`[colab] Started background exec (pid=`),
or read the envelope. The same mistake in three other places that day: a test
asserting a poll loop was bounded matched the word `MAX_WAIT` in the recipe's
own failure message; a test forbidding `sys.path` use matched the docstring
explaining why the file avoids it; and a regex deriving GPU recipes matched
`exec ` with a trailing space, silently excluding every `exec-async` one.
**Match structure, not prose.**

## 7. The pre-flight that makes most of this unnecessary

`exec -f` sends a file's *text*. Nothing it imports travels with it. So
reproduce the VM's import environment locally, in about a second, before
provisioning anything:

```bash
mkdir /tmp/deploycheck && cp <every file you upload> /tmp/deploycheck/
cd /tmp/deploycheck && python -c "
import sys; sys.path.insert(0, '/tmp/deploycheck')
import importlib
for m in ('mod_a', 'mod_b', 'mod_c'):
    try: importlib.import_module(m); print('OK  ', m)
    except Exception as e: print('FAIL', m, type(e).__name__, e)
"
```

A bare directory containing only the uploaded files *is* `/content`. If an
import fails there, it fails on the VM — after you have paid for an A100 and
pushed 242MB.

## 8. Fix deployability at the source, not with more freight

When a module fails to import remotely because it assumes the repo layout,
the tempting fix is to ship the layout: tar the package, upload, extract.
That works and is wrong — it moves 148KB and five unrelated modules to
satisfy imports the remote job never calls.

Prefer a deployable fork containing only what the job needs, with no repo
layout in it, **pinned by a test** that asserts its functions are
bit-identical to the originals. Duplication is a hazard; an unchecked
duplicate is the hazard, and a checked one is a deployment artifact.

---

**The short version.** Read the log, all of it, ANSI stripped. Trust the
traceback over your memory of the code. Remember it shows the first failure
only. Ask the envelope whether the job finished, because a killed job says
nothing. Confirm the log belongs to this run. And spend the one second on a
local import check that a GPU provision costs you.
