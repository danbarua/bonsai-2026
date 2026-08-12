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

### 1.1 `upload` is slower than GCS, but by less than we first claimed

Measured from one machine, minutes apart, on identical bytes:

| leg | path | bytes | time | rate |
|---|---|---|---|---|
| 0 | local -> GCS | 250.6 MB | 123.2s | 2.03 MB/s |
| 1 | local -> VM, `upload` x12 chunks | 242.4 MB | **205s** | **1.18 MB/s** |
| 2 | **GCS -> VM** | 250.6 MB | **4.1s** | **61.5 MB/s** |

**`upload` runs about 1.7x slower than the same box reaches GCS** -- a real
gap but a modest one, and both are near this uplink's ceiling. The story is
not that `upload` is slow.

**The story is leg 2: in-cloud transfer is 52x faster than anything from the
caller's building.** Which makes the cost model arithmetic:

    direct:    205s x N sessions
    via GCS:   123s once, then 4.1s x N

GCS wins at N=1 (127s vs 205s) and the gap widens with every session. Four
launches of one job today cost ~14 minutes of uploading; staged once it is
2.3 minutes. Leg 2 was verified on arrival, not merely timed: exact byte
count, and the payload unpickles to the expected array shape.

**A correction, recorded because the first version of this document got it
wrong.** An earlier draft claimed 0.18-0.28 MB/s and a 7-10x gap. Those
figures came from differencing `mighty-colab log` session timestamps, whose
semantics we had assumed rather than checked, across sessions where other
work may have been in flight. The direct measurement above supersedes them.
Inferring throughput from event timestamps in a log not designed to measure
it produced a number off by a factor of five.

Cost in practice: **~3.5 minutes of upload per session**, paid again on every
relaunch, guarding 118s of evolution and ~2 minutes per CV arm. Four launches
of the same job spent roughly fifteen minutes pushing identical bytes.

The chunking itself is a workaround already in this repo's history: a single
250MB pickle exceeded the transfer endpoint's size limit, so the driver splits
into twelve 20MB `.npy` files and reassembles on the VM.

### 1.2 Everything else was fast and correct

- Session provision to READY: **~10s**.
- `reinstall` of `jax[cuda12] diffrax equinox optax`: **~10s** via uv.
- GPU evolution of 60,000 images x 4 topologies: **117.7s**, reproducing a
  months-old run to within 3.3% across three separate A100 allocations.
- The whole ten-arm job: **1,321.8s (22 minutes), 10/10 converged**, against
  **124,253.2s = 34.51 core-hours** for the same arms on local CPU. Stated on
  both bases, because they differ by 2% and the document should say which:
  **95.9x** comparing summed arm time to summed arm time (1,296.1s), and
  **94.0x** comparing it to total job time (1,321.8s, which includes feature
  building between arms). Per-arm speedups ranged 69x to 183x.
- One `exec-async` submission, one clean teardown, `status=ok` with the
  driver's sentinel present, results downloaded.

The job that motivated all of this therefore ran in less time than the four
failed launches spent uploading.

### 1.3 A three-way factorial the CLI made cheap

An external reviewer asked whether a GPU/CPU numerical difference (A100 TF32)
could explain a divergence we had attributed to a convergence criterion. The
question is answerable only by running the same code on a different backend,
which is one environment variable here:

    JAX_PLATFORMS=cpu python disambiguate_jax_sklearn_divergence.py

| fit | accuracy | iterations |
|---|---|---|
| sklearn CPU | 0.882250 | 5,309 |
| JAX GPU | 0.871083 | 1,955 |
| JAX CPU | 0.871000 | 2,003 |

JAX-on-CPU is **135x closer to JAX-on-GPU than to sklearn**, ruling out the
platform. Worth recording as a usage pattern: keeping the remote driver and
the local one importing the *same* module made this a one-line experiment
rather than a port.

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

### 3.6 A Colab VM has no service account — **cost: one A100, one leg**

`storage.Client()` on the VM falls through to the GCE metadata service and
raises, which is easy to miss when the same line works locally under ADC:

```
RefreshError: ("Failed to retrieve
http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/
... Status: 404", ...)
```

Two working shapes, depending on the object:

```python
# public-read object: no credentials needed, none shipped
client = storage.Client.create_anonymous_client()

# private object: upload the key, point BONSAI_GCS_CREDENTIALS at its
# REMOTE path, and pass it with --env
client = storage.Client.from_service_account_json(os.environ["BONSAI_GCS_CREDENTIALS"])
```

The trap is that `--env BONSAI_GCS_CREDENTIALS=/some/path` names a path on
the **VM**, so setting it without also uploading the key produces a confusing
"file not found" one layer further in than the real mistake.

### 3.7 Our own bugs, recorded because they are agent-shaped

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
- **Deriving a throughput number from session-log timestamps.** We reported
  0.18-0.28 MB/s from differencing `FILE: upload` events, and a direct
  wall-clock measurement of the same operation gave 1.18 MB/s. The log is an
  event record, not an instrument; timing something means timing it.

---

## 4. Requests, ordered by measured cost

**Prior art in this repo, found after the fact.** Stage 2B's `DESIGN.md:507`
already locks the rule these measurements argue for: artifacts are "pushed to
Google Cloud Storage from within [the cloud environment] -- never
round-tripped through local upload (Stage 2A's 242MB-vs-~6-15MB Colab upload
limit, already hit once)." That rule binds Stage 2B; this was Stage 2A work,
which is where the limit was hit and the lesson learned. So this document is
not a rule violation -- it is the same conclusion re-derived independently on
the other side of the boundary, with the number the original never had: 52x.

1. **Let the VM fetch from object storage instead of from the caller.**
   This is the whole ballgame, and it is measured: 205s from here versus 4.1s
   from GCS for the same bytes, a 52x difference that no amount of tuning the
   caller-side path can close. `mighty-colab upload --from-gcs gs://...`, or
   a documented VM-side helper, removes the slowest step in every
   repeated-job workflow. We are doing it by hand today with
   `google-cloud-storage` in the driver.
   *Traces to: §1.1 (205s vs 4.1s, measured three ways); §3.6 (the VM has
   no service account, so the fetch path needs an anonymous or key-based
   client); Stage 2B `DESIGN.md:507`, which locked this rule already.*
2. **Cross-session reuse of identical bytes.** Even without object storage,
   the same 242MB is re-uploaded per session with no caching. Content-
   addressed reuse would pay for itself on the second launch. Raw throughput
   is NOT the ask -- at 1.18 vs 2.03 MB/s the endpoint is close enough to
   the uplink that tuning it buys little.
   *Traces to: §1.1 -- four launches of one job re-uploaded identical bytes,
   ~14 minutes against 2.3 minutes staged once.*
3. **Don't clobber a previous run's log/sidecar.** Either timestamp by
   default, or refuse to overwrite a non-empty `--output-log` without a
   `--force`. The sidecar especially: it is the durable failure record and it
   is one relaunch from being lost.
   *Traces to: §3.2 (one false diagnosis, 13:42 against a 13:35 log) and
   §3.3 (a deleted log recovered from its sidecar, which the next run would
   have overwritten).*
4. **An import pre-flight for `exec -f`.** Something like
   `mighty-colab exec --check-imports -f driver.py` that reports which
   top-level imports would not resolve on a bare VM. This is our most
   expensive recurring failure and it is statically detectable.
   *Traces to: §3.1 -- two A100s and ~40 minutes, twice, before a
   one-second local import check would have caught it.*
6. **Give `new`'s assign failure something an agent can decide on.** Six
   concurrent A100 provisions drew 503s; a single retry minutes later
   succeeded; two further singles failed again. All we get is:

   ```
   Failed to issue request POST .../tun/m/assign?...&accelerator=A100:
   Service Unavailable
   ```

   No body, no `Retry-After`, no distinction between transient capacity,
   a burst-triggered backoff, and an exhausted daily quota — which are three
   different decisions (retry soon / back off / stop for the day). Google
   serves HTML on these, so a body very likely exists and is discarded.
   Note the asymmetry: keep-alive errors DO carry the raw `response_body`
   (per the skill docs), while the one error that gates every run does not.
   Capturing it, and populating the envelope's `hint` with a
   capacity-vs-quota reason, would turn a dead end into a retry policy.
   *Traces to: §5 -- three failed provisioning attempts, no diagnostic
   signal, and a partial experiment abandoned as a result.*

5. **Raise or document the upload size limit** that forces 250MB into twelve
   chunks. The workaround is in three drivers here.
   *Traces to: §1.1 and Stage 2B `DESIGN.md:507`, which cites the same
   limit as its reason for going GCS-native.*

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
