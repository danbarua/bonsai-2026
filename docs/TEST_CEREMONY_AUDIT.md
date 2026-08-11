# Test ceremony audit — blocking guards under `tests/`

## What this is

An external audit of this repository's blocking test guards, run
2026-08-11 against branch `stage2b`, asking of each: does it name a harm,
does it fire when that harm occurs, does the fix fork on the verdict, and
does it pin an outcome or a description? Its ledger is git — commits that
moved each pinned literal, and whether any of them named a real defect.

The audit's own framing: *"Deletion is Dan's call; this file is evidence
only."* That ruling was given the same day and is recorded under
DISPOSITION below. The audit text is kept verbatim as evidence; nothing
in it has been edited to match the outcome.

The companion catalogue is `docs/VACUOUS_TESTS.md`, which asks a
different question: whether a test passes for reasons unrelated to what
it names. A test can be perfectly non-vacuous — firing precisely when its
predicate says it should — and still be pure ceremony, because the
predicate itself measures something nobody needs measured. This document
covers that second failure.

## Disposition, 2026-08-11

Dan ruled DELETE on all three of the audit's named offenders. Landed:

- `test_each_exemption_still_contributes_the_candidate_count_it_did`
  deleted (171 lines, of which ~120 were a changelog of its own re-pins),
  replaced by `test_the_exempt_set_is_exactly_these_documents` — the same
  protection against silent drift of WHICH documents are exempt, with no
  integers.
- `test_the_corpus_derives_the_scoped_candidate_count`'s `== 89` dropped;
  the test survives as an anti-vacuity floor, because deriving NOTHING is
  a real and different failure that would let the disposition check pass
  over an empty set.
- The inventory-kind test's `len(answered) == 89` dropped for the same
  reason, keeping per-kind non-emptiness and the anti-vacuity.

The final ledger at the time of deletion, re-derived from git rather than
taken from the audit's snapshot — which had gone stale twice during its
own review, the pins moving again while the report recommending their
removal was being written:

**15 commits moved a pinned count.** `README.md` travelled
21 → 20 → 21 → 22 → 23 → 24 → 25 → 26 → 28 → 29; `FINDINGS.md` travelled
37 → 46 → 52 → 59 → 60 → 61 → 63. **None** named an obligation smuggled
into an exempt document.

A batching fix — reporting every drifted count in one run instead of the
first — was built before this audit was read, and is deleted with the
test it was optimising. Reducing the latency of a check that had never
been right was the wrong repair, and the evidence for deleting it was
already in hand when the repair was proposed.

## What the audit establishes beyond the three deletions

The census matters as much as the verdict. Forty-eight blocking guards
matched the pre-declared ceremony predicate; **one** was deleted and two
narrowed. The rest fork on substance and stay. Science and system pins —
`ENCODER_STEPS == 1200`, `REF_IDX == 363`, `N = 60,000`, feature dimension
1008 — were enumerated and then explicitly excluded, because they fail
when the EXPERIMENT changes meaning and the fix is never "bump the test".

So the cost was never spread across the suite. It concentrated in one
test body, on the same commits that closed real science, which is exactly
where it does the most damage: the operator who has just spent attention
re-justifying a module-map sentence is the operator who must not
rubber-stamp the next `allow_test_split` failure.

---

# The audit, verbatim

**Branch:** `stage2b`
**Date:** 2026-08-11
**Scope:** blocking tests under `tests/` (CI / `make test` / `make stage2b-test`).
**Not this audit:** vacuity (`docs/VACUOUS_TESTS.md`); tooling under `tools/` except as imported by tests.
**Method:** four discriminators + git `-S`/`-G` re-pin ledger. Deletion is Dan's call; this file is evidence only.

**Calibration specimens (already adjudicated):**

| Specimen | Status | Why |
|---|---|---|
| `test_artifact_json_contract.py` | DELETED (`8a2a90d`) | Zero candidates for the harm named; one self-inflicted stoppage |
| `test_allow_test_split_true_appears_in_exactly_this_driver_plus_the_exemptions` | KEEP | Fires on new test-split readers; fix forks on substance |
| `test_each_exemption_still_contributes_the_candidate_count_it_did` | LIVE CEREMONY | Hand-pinned clause counts; fix is always bump + comment |

## Candidate count (before classification)

**48** distinct blocking tests (or coherent one-assert groups) matched the pre-declared predicate:

- hand-pinned integer counts / equality pins on **document or inventory shape**, or
- set/frozenset equality over **filenames, exemptions, allowlists, PROTOCOL_DOCS**, or
- enumeration guards whose green path is "the list still looks like the list."

Science-outcome pins (e.g. `EXPECTED_REF_IDX == 363`, `ENCODER_STEPS == 1200`, `N=60000`, feature dim 1008, pairwise `len==6`) were **enumerated then excluded** from the ceremony candidate set — they pin system constants that break science if wrong, not cartography of records.

48 ≥ 10 → predicate not widened.

**Blocking surface size (context, not candidates):**

| surface | test functions | lines |
|---|---:|---:|
| all `tests/` | 1 253 | 24 150 |
| `STAGE2B_TEST_FILES` (24 files) | 853 | 14 715 |

## Verdict summary

| test | verdict |
|---|---|
| `test_each_exemption_still_contributes_the_candidate_count_it_did` | **DELETE** |
| `test_the_corpus_derives_the_scoped_candidate_count` (`== 89`) | **FIX** — drop the bare count, keep membership tests |
| `test_the_inventory_still_holds_a_disposition_for_every_kind` (`total == 89`) | **FIX** — assert each kind non-empty, drop the count |
| `test_no_clause_in_the_real_corpus_is_left_undispositioned` | KEEP — real structural guard |
| `test_every_document_on_disk_is_corpus_or_named_exemption` | KEEP — membership, not counts |
| `test_every_declared_document_still_exists` | KEEP |
| `test_the_corpus_and_the_exemptions_do_not_overlap` | KEEP |
| `test_every_exemption_carries_a_substantive_reason` | KEEP |
| `test_every_binds_at_pointer_names_a_clause_in_a_binding_kind` | KEEP |
| `test_the_prose_states_no_count_of_itself` (`tests/test_catalogue_counts.py`) | KEEP — forbids the stale shape rather than pinning N |
| `test_the_incident_numbers_are_contiguous` | KEEP |
| `test_allow_test_split_true_appears_in_exactly_this_driver_plus_the_exemptions` | KEEP — fires only on a new reader of the locked split |
| `test_every_exemption_still_refers_to_something_real` | KEEP |
| `test_the_stage2b_test_target_lists_every_stage2b_test_file` | KEEP — principle 21 narrowing completeness |
| CI allowlist disjoint from spending (`tests/test_cloudbuild_ci.py`) | KEEP |
| `test_every_cited_path_exists` (`tests/test_doc_references.py`) | KEEP |
| Stale-exemption twins (`tests/test_ci_image_dependencies.py`, `tests/test_hook_registrations.py`) | KEEP |
| `test_every_stats_condition_has_exactly_one_path_segment_and_back` | KEEP — bijection |
| `test_every_cited_test_function_still_exists` (`tests/test_stage2b_negative_path_evidence.py`) | KEEP |
| Fingerprint / GCS / ridge / ladder science and transport pins | KEEP — out of ceremony set |
| `tests/test_gate_inventory.py` synthetic fixture pins | KEEP — mechanism |
| `tests/test_tools_conventions.py` floors | KEEP — anti-vacuity |
| `tests/test_stage2b_artifact_manifest.py` label uniqueness | KEEP |

## Precision ledger — ceremony core

Pinned dict at audit time:

```text
FINDINGS.md: 60
NEGATIVE_PATH_EVIDENCE.md: 19
PHASE_B_PLAN.md: 38
README.md: 26
```

Re-pin commits (false positives — each a bump after an honest narrative or module-map edit):

| SHA | Effect |
|---|---|
| `bff25eb` | Introduced pin |
| `476a7ed` | Re-confirm README after own edit |
| `36e44b0` | FINDINGS 37→46 |
| `5fb1a7a` | README 20→21 |
| `3d16f1a` | README 21→22 (Protocol 2 module-map) |
| `207a677` | FINDINGS 46→52 (audit result write-up) |
| `f3b140a` | README 22→23 (manifest module-map) |
| `bb95666` / `b5f3db1` | README 23→24; FINDINGS 52→59 (Protocol 1) |
| `1771897` | FINDINGS 59→60 (companion closure reword) |
| `90410e4` | README 24→25 (scaler Lipschitz module-map) |
| `12d4bf1` | README 25→26 (combined-operator-norm sentence) |

**True positives naming a smuggled binding obligation: 0.**

Every firing's fix was identical in shape: re-derive the count, paste the
integer, paste a multi-paragraph "Re-read: still not binding" comment.

**Alarm fatigue.** This test trained a reflexive "bump the exemption
counts" loop on the same commits that close real science. Operators who
have just re-justified module-map NEVERs are the same operators who must
not rubber-stamp a real `allow_test_split` or undispositioned-clause
failure next.

**Realistic path to harm if removed:** an exempt document could grow a
sentence that is a true new MUST in its own voice, and nothing would count
its clause total. **Why acceptable:** exempt documents are explicitly
non-binding; binding force lives in `PROTOCOL_DOCS` and `gates.toml`,
already covered by `test_no_clause_in_the_real_corpus_is_left_undispositioned`.
A count pin does not detect "became binding" — it detects "word count of
MUST-shaped English changed", which is what FINDINGS does every run.

## Single highest-value fix

Remove hand-pinned clause totals (`89`, and per-exempt 60/19/38/26); keep
structural inventory guards. Replace the deleted test's body with the
exempt-set membership assertion and nothing else. Do **not** add a new
test whose job is to justify the deletion.

Expected effect: science close-outs stop failing `make stage2b-test` for
cartography; real fires keep full severity.

## Limits

- Did not re-run the full suite; classification is static plus git history.
- Some of the 48 are stale-exemption twins, counted once in spirit.
- Findings advise; they do not gate merges.
