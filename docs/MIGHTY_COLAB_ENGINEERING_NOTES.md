# mighty-colab 0.4.1: engineering notes from a real run

Handoff notes for `mighty-colab` development, from driving it hard on
2026-08-12: one long GPU job (ten cross-validation arms over 60,000 images,
four topologies) plus a transfer benchmark, across five A100 sessions and
four launches of the same job — three of which failed, which is where most of
the value in this document comes from.

Written engineering-first. The science this produced is in
`GAUGE_COMPARISON_2A_RESULT.md`; nothing here depends on it.

All logs preserved under
`experiments/stage2a_dynamics_classification/results/gauge_comparison_2a/previous_runs/`,
log and `.json` sidecar per run, timestamped by the log's own mtime.

---

## 1. Headline measurements

### 1.1 `upload` throughput is the dominant cost, by an order of magnitude

Measured from one machine, minutes apart, on identical bytes:

| path | bytes | time | rate |
|---|---|---|---|
| `mighty-colab upload`, 12 x 20MB chunks | 242.4 MB | 1,266s | **0.18 MB/s** |
| `mighty-colab upload`, second session | 202.0 MB | 731s | **0.28 MB/s** |
| local -> GCS (`google-cloud-storage`) | 250.6 MB | 123.2s | **2.0 MB/s** |

**`upload` runs 7-10x slower than the same box reaches GCS.** This is not the
uplink: the GCS leg saturates at 2.0 MB/s from the same connection in the same
period. Per-chunk gaps for identical 20MB files ranged 60s to 145s, which
reads as per-call overhead rather than steady streaming.

Cost in practice: **~20 minutes of upload per session**, paid again on every
relaunch. Four launches of one job spent roughly an hour pushing the same
242MB. The compute it guards is 118s of evolution and ~2 minutes per CV arm.

The chunking itself is a workaround already in this repo's history: a single
250MB pickle exceeded the transfer endpoint's size limit, so the driver splits
into twelve 20MB `.npy` files and reassembles on the VM.

### 1.2 Everything else was fast and correct

- Session provision to READY: **~10s**.
- `reinstall` of `jax[cuda12] diffrax equinox optax`: **~10s** via uv.
- GPU evolution of 60,000 images x 4 topologies: **117.7s**, reproducing a
  months-old run to within 3.3% across three separate A100 allocations.
- CV arm on A100 vs the same arm on local CPU: **125.3s vs 10,162s (81x)**.

---

## 2. Patterns that worked

### 2.1 Three-way session guard, not two

`new` provisions unconditionally, so a naive "if not found, create" branch
allocates a second billable VM whenever `status` cannot answer — expired
credentials, a network drop, a backend 5xx. The envelope separates the cases:

```bash
st=$(mighty-colab --json status -s "$SESSION" 2>/dev/null)
sst=$(printf '%s' "$st" | jq -r '.status // "malformed"')
srsn=$(printf '%s' "$st" | jq -r '.reason // ""')
if   [ "$sst" = "ok" ];                   then echo "reusing $SESSION"
elif [ "$srsn" = "session_not_found" ];   then mighty-colab new -s "$SESSION" --gpu A100
else echo "REFUSING: cannot determine whether '$SESSION' exists"; exit 1
fi
```

`reason=session_not_found` is the absent case *specifically*. Anything else
non-`ok` is refused loudly. This is the single most useful thing `--json`
enabled for us.

### 2.2 Teardown checks BOTH the exit code and the envelope

Neither alone is sufficient. A non-zero exit catches the CLI failing to
complete its transaction at all (no envelope to read); a non-`ok` status
catches a completed transaction reporting that teardown itself failed.

```bash
src=0
sout=$(mighty-colab --json stop -s "$SESSION") || src=$?
sreason=$(printf '%s' "$sout" | jq -r '.status // "malformed"')
if [ $src -ne 0 ] || [ "$sreason" != "ok" ]; then
  echo "LEAK WARNING: teardown of '$SESSION' exited $src (status=$sreason)"
fi
```

`stop` on an absent session returning `already_stopped` with rc=0 is exactly
right for unconditional teardown — please keep it.

### 2.3 `exec-async` submission is verified, not assumed

```bash
aout=$(mighty-colab --json exec-async -s "$SESSION" -f driver.py \
        --timeout 3600 --output-log "$LOG") || rc=$?
astat=$(printf '%s' "$aout" | jq -r '.status // "malformed"')
[ "$astat" = "started" ] || { echo "did not start (status=$astat)"; exit 1; }
```

Without the status check, a refused submission is followed by polling an
empty log until the ceiling.

### 2.4 Completion comes from the envelope, not from grep

```bash
off=0; jstat=running
while [ $waited -lt $MAX_WAIT ]; do
  tout=$(mighty-colab --json log -s "$SESSION" --tail --since-offset $off)
  jstat=$(printf '%s' "$tout" | jq -r '.status // "malformed"')
  printf '%s' "$tout" | jq -r '.text // empty'
  off=$(printf '%s' "$tout" | jq -r '.next_offset // empty')
  [ "$jstat" != "running" ] && break
  sleep 30; waited=$((waited + 30))
done
```

`worker_terminated` is the case that justifies this. **An OOM kill or a
backend teardown prints no traceback**, so a loop that decides completion by
grepping the log for a sentinel or `Traceback` sees nothing and waits out its
timeout against a dead process. We hand-rolled the grep version first and
replaced it.

The driver's own sentinel is still checked, and is not redundant: `status=ok`
means the job did not raise, which cannot distinguish "ran to completion and
passed its gate" from "exited cleanly before reaching it".

### 2.5 Sync for seconds, async for hours

`exec-async` on a 115-second evolve adds a poll loop around a call that
returns before the loop's first sleep. We use plain `exec` for the
seconds-scale GCS-download benchmark and `exec-async` only for the multi-hour
CV job. The distinction is worth stating in the docs, because the async path
is otherwise tempting everywhere.

---

## 3. Pitfalls, with what each one cost

### 3.1 `exec -f` sends text; imports do not travel — **cost: 2 A100s, ~40 min**

The documented behaviour, encountered anyway. Our driver imported
`stage2a_core`, which does this at module scope:

```python
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "stage1d_topology_specificity"))
from build_stage1d_constructions import build_and_verify_T
```

On the VM those paths resolve to nothing. The job died after provisioning and
after a full 242MB upload — twice, because the first fix addressed the module
the error named rather than the class of problem.

**Fix at the source.** We forked the deployable subset (`stage2a_core_colab`,
pure numpy, no repo layout) and pinned it to the original with a test
asserting bit-identical outputs. The rejected alternative was tarring the
148KB package closure and extracting it on the VM — it works, and it ships
five unrelated modules to satisfy imports the job never calls.

**The pre-flight that makes this free**, which we should have run first:

```bash
mkdir /tmp/deploycheck && cp <every file you upload> /tmp/deploycheck/
cd /tmp/deploycheck && python -c "
import sys; sys.path.insert(0,'/tmp/deploycheck')
import importlib
for m in ('mod_a','mod_b','mod_c'):
    try: importlib.import_module(m); print('OK  ', m)
    except Exception as e: print('FAIL', m, type(e).__name__, e)
"
```

A bare directory containing only the uploaded files *is* `/content`.

### 3.2 `--output-log` is a fixed path nobody clears — **cost: one false diagnosis**

A relaunch that fails early leaves the previous run's bytes in place. Two
consequences:

- a monitor tailing from byte zero replays an old traceback as live (we
  diagnosed a 13:42 run against a log written at 13:35);
- **a success check that greps for a sentinel matches the previous run's
  success.** A job that wrote nothing at all passes.

The obvious fix — truncate at submit — destroys the most valuable artifact a
failed run produces. We archive instead, taking the sidecar along:

```bash
if [ -s "$LOG" ]; then
  ts=$(date -r "$LOG" +%Y%m%dT%H%M%S)
  mkdir -p "$(dirname "$LOG")/previous_runs"
  mv "$LOG" "$(dirname "$LOG")/previous_runs/$(basename "$LOG").$ts"
  [ -f "$LOG.json" ] && mv "$LOG.json" ".../previous_runs/$(basename "$LOG").$ts.json"
fi
```

### 3.3 The `.json` sidecar is excellent, and silently overwritten

`<log>.json` surviving `stop` is the best diagnostic surface in the tool. It
carries `status`, `exit_code`, `reason`, and `blocks` — including the prelude
verbatim:

```json
{"code": "import sys\nsys.argv = ['run_gauge_comparison_2a_gpu.py']\n__name__ = '__main__'\n__file__ = '<mighty-colab-exec:run_gauge_comparison_2a_gpu.py>'\n..."}
```

A log deleted by hand was **reconstructed in full** from it:

```bash
jq -r '.blocks[]?.outputs[]? | (.text // .traceback // empty)
       | if type=="array" then join("") else . end' "$LOG.json"
```

But the next run writes the same path, so the record of why the previous run
failed is one relaunch from gone.

### 3.4 `hint` and `message` are easy to drop

We read `status`, `exit_code` and `reason` and ignored `hint`/`message` for
most of the day. `reason` is a code; `hint` is where the CLI says what to do.
Both are printed on failure now, next to the log and sidecar **paths** — a
failure report that describes the evidence without naming its location makes
the reader go find it.

### 3.5 The `__file__` prelude is why any of this was diagnosable

Worth stating because it is invisible when it works. The traceback reads:

```
<mighty-colab-exec:run_gauge_comparison_2a_gpu.py> in <cell line: 0>()
---> 41 import stage2a_core as s2a
/content/stage2a_core.py in <module>
---> 30 from build_stage1d_constructions import build_and_verify_T
```

Exact file and line at both levels. Without the prelude the top frame is
`<ipython-input-3-...>` against nothing. Please keep it.

### 3.6 Our own bugs, recorded because they are agent-shaped

Not tool defects — but they are the kind of mistake an agent driving this CLI
will make, so they may be worth designing against:

- `pgrep -fc` is not supported on macOS; it exited rc=2, our `|| echo 0`
  fallback read that as "no processes", and a monitor declared a healthy run
  dead while 20 processes were fitting.
- `make` echoes each recipe before running it, so a watcher gating on
  `grep "started as pid"` matched the `echo` **inside the recipe** and fired
  before anything was submitted. Gate on strings only the CLI emits
  (`[colab] Started background exec (pid=`), or on the envelope.
- Backticks inside a `git commit -m "..."` message executed as command
  substitution, because the message quoted `log --tail --json`.

---

## 4. Requests, ordered by measured cost

1. **Upload throughput.** At 0.18-0.28 MB/s this dominates every run we do,
   and it is 7-10x below what the same connection achieves to GCS. Even
   parallelising the per-chunk calls would help; a bulk/multi-file upload
   would help more.
2. **A GCS (or any object-store) fetch path.** The natural shape for a
   repeated job: stage the input once, have each VM pull it in-cloud. We are
   doing this by hand with `google-cloud-storage` inside the driver plus
   `--env` credentials. `mighty-colab upload --from-gcs gs://...` (or a
   documented VM-side helper) would remove the slowest step entirely.
3. **Don't clobber a previous run's log/sidecar.** Either timestamp by
   default, or refuse to overwrite a non-empty `--output-log` without a
   `--force`. The sidecar especially: it is the durable failure record and it
   is one relaunch from being lost.
4. **An import pre-flight for `exec -f`.** Something like
   `mighty-colab exec --check-imports -f driver.py` that reports which
   top-level imports would not resolve on a bare VM. This is our most
   expensive recurring failure and it is statically detectable.
5. **Raise or document the upload size limit** that forces 250MB into twelve
   chunks. The workaround is in three drivers here.

---

## 5. Preserved logs

```
experiments/stage2a_dynamics_classification/results/gauge_comparison_2a/
  gpu_run.log                    # live run
  gpu_run.log.json               # its sidecar
  previous_runs/
    gpu_run.log.20260812T133544        # ModuleNotFoundError run
    gpu_run.log.20260812T133544.json   # its sidecar (11,945 bytes)
```

The 13:35 log was deleted by hand and recovered from its sidecar; both are
committed. Session-level history for all five sessions is available via
`mighty-colab log -s <session>` and is quoted in section 1.1.
