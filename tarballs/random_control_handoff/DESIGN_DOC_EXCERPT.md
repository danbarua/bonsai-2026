# Excerpt from bonsai_diffusion_stage0_locked_design.md (original design lock)

Relevant section, verbatim, establishing the normalization procedure applied
to ALL FOUR graph constructions (T, rewired, random, lattice), not just random:

---

**2. Coupling normalization applied before any comparison.** Mean
weighted degree varied substantially before correction (T: 3.881,
rewired: 3.881 -- identical by construction, matched-sparsity random:
2.038, lattice: 3.703). All four were rescaled to a common budget
(C = T's own mean weighted degree, 3.881) by A_tilde = A * (C / mean
weighted degree). Rewiring's exact match to T pre-normalization confirms
it preserves weight assignment as closely as possible, as required.

---

Later, in the calibration panel section:

---

- Graph normalization: equal mean weighted degree (C = 3.881, T's own
  value, applied identically to all four constructions) -- this controls
  the global coupling budget, not edge-level coupling strength; the
  lattice's fewer edges under equal total budget necessarily means
  stronger per-edge coupling, which is a documented property of this
  normalization choice, not a flaw to correct now

---

And in "Reproducing these results":

---

All four constructions built from KMNIST class 0's topology and its
active-node set, normalized to equal mean weighted degree, saved to
`kmnist_c0_controls_normalized.npz`.

---

# What this establishes

- The mean-weighted-degree normalization was a DELIBERATE, DOCUMENTED
  design choice (explicitly labeled "Correction 2, applied per review"),
  not an incidental byproduct of some other process.
- It was applied identically to all four constructions, not something
  specific to "random."
- The formula is: A_tilde = A * (C / mean_weighted_degree(A)), where
  C = T's own mean weighted degree (3.881 for KMNIST class 0).
- The RAW (pre-normalization) random construction's mean weighted degree
  was 2.038 -- this is the value BEFORE the rescaling in kmnist_c0_controls.npz.

# What was independently verified against class0_constructions.pkl

- kmnist_c0_controls_normalized.npz's 'random' key has mean weighted
  degree 3.8811482995463593 -- matches class0_constructions.pkl's cached
  'random' construction's mean weighted degree to FULL FLOAT64 PRECISION.
- Edge count: 1104 (in the normalized npz) vs. 1090 (in class0_constructions.pkl)
  -- a small discrepancy, most likely a different RNG seed/state for the
  raw edge-placement draw, not a different algorithm, given the
  normalization target matches exactly.
- Structural check on the RAW (pre-normalization) random construction:
  - Only 8 of its 1104 edges coincide with T's own edge positions --
    consistent with independent random placement, not derived from T's
    specific structure.
  - Edge count ratio to T: 1104/2102 = 0.5252... -- approximately but
    not exactly half; the EXACT rule generating this count is not yet
    known and needs recovery (see caution below).
  - The edge WEIGHT VALUES used (e.g. 0.90016328, 0.90040547) overlap
    with T's own weight-value pool -- suggesting weights are drawn from
    T's own weight distribution/pool, not independently generated.

# Open question needing recovery, not assumption

"Roughly half T's edge count" is not yet a confirmed exact rule. Candidates
to check (not exhaustive):
- floor(|E_T| / 2) or similar simple arithmetic rule
- A Bernoulli draw over some candidate edge set with p chosen to yield
  ~50% density
- A rule tied to the active-node count or ink_mask rather than |E_T| directly
- Something else not yet considered

Recover this from the code/metadata/RNG traces if at all possible, or by
testing candidate rules against the exact 1104 (raw npz) and 1090 (final
pkl) edge counts -- don't infer the rule from one data point and assume
it's right.
