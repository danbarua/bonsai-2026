# Methodology Synthesis: What This Line of Work Actually Discovered

## The evidence hierarchy, redrawn

| Claim | Status |
|---|---|
| Additional representational capacity exists beyond T | Established |
| E contains unique linearly non-recoverable information | Established |
| E survives methodological stress-testing | Established |
| R contributes unique information | Suggestive |
| R is robust to methodological stress-testing | Not yet established |

This replaces the earlier, looser "strong/medium/weak" framing. The
distinction that matters: E and R are not two results at different
confidence levels on the same claim -- they are qualitatively different
positions in the hierarchy, and the gap between "established" and
"suggestive" is exactly what the regularization-hygiene check was for.

## The precise epistemic distinction to hold going forward

Not: *E is the important channel.* Instead: **E is the channel whose
importance is currently demonstrated most robustly.** Science accumulates
evidence while systematically removing alternative explanations; it does
not prove importance outright. Every claim in this document set should be
read under that framing -- including this one, which is itself a claim
about the current state of evidence, not a final word.

## Why the E/R asymmetry is the actual finding, not a disappointment

The original expectation was roughly symmetric channels. They aren't:

```
T
├── E   -- robust: survives residualization, correspondence destruction,
│          dependence analysis, matched-dimensional random controls,
│          regularization tuning, and ridge-strength variation, with the
│          effect strengthening (not weakening) under the hygiene check
│
└── R   -- useful, directionally consistent, but its one significant
           result did not survive the same hygiene check
```

Had both channels survived every test identically, that would have been
a reason for more suspicion, not more confidence -- real, independent
signals are not expected to behave identically under increasingly
adversarial tests. The methodology discriminating between E and R is
itself evidence the methodology is working, not a partial failure of the
R hypothesis.

## The reusable methodology

```
Stage 1: Search for additional representational capacity.
Stage 2: Destroy the result using progressively better null models.
Stage 3: Decompose successful representations.
Stage 4: Measure dependence between the decomposed parts.
Stage 5: Residualize to isolate unique information.
Stage 6: Audit robustness of every remaining claim against analysis
         choices (regularization, hyperparameters, residualization model).
Only then interpret mechanism.
```

Two of these stages are easy to conflate and shouldn't be:

- **A null model** asks: could something simpler explain this? (Stage 2 --
  duplication controls, shuffled controls, random-projection ensembles,
  matched-dimensionality comparisons.)
- **A hygiene audit** asks: could my analysis pipeline itself explain
  this? (Stage 6 -- does the result survive regularization tuning,
  different residualization strengths, different train/calibration
  splits?)

These catch different failure modes. A result can cleanly survive every
null model in Stage 2 and still be an artifact of Stage 6 -- exactly what
happened to R's conditional result here: it beat duplication, shuffling,
and random-projection ensembles, and only weakened when the
regularization-hygiene audit was applied. Neither stage is a substitute
for the other. **Every surviving positive result deserves a hygiene
audit, not just a null model** -- this is offered as a general principle
for this project's future work, not a one-off fix for this specific
finding.

## What the project has become

The organizing questions changed over the course of this work, and the
change itself is worth naming: not "did accuracy improve" but "what
information is represented, which channels are robust, how redundant are
they, and which survive increasingly hostile tests." That is
representation discovery, not feature engineering -- and it is a
different, harder standard than the one this line of work started under
when Capacity Experiment III first found a positive accuracy delta.

## Honest scope of this document

This is a synthesis and a methodological formalization, not a new
experimental result -- it makes no claims that weren't already
established or qualified in the documents it draws together. Its value is
organizational: stating the evidence hierarchy and the six-stage
methodology explicitly, so that future extensions of this work (the
global ink control, frequency-band ablation, and any further conditional
residualizations) can be slotted into the stage where they belong, and
judged by the standard that stage implies -- a null-model question is not
answered by a hygiene audit, and a hygiene audit is not answered by a
null model.
