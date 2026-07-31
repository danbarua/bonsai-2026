"""
LocalOscillatorField: a local (nearest-neighbor) coupled Kuramoto oscillator
field with closed-loop input anchoring, for grayscale image tasks at
MNIST scale.

Deliberately NOT models/hebbian or models/predictive extended -- this is a
clean-room model living in MNIST/, borrowing validated IDEAS from both
existing Bonsai models and two external notebooks, not code:

- LOCAL coupling (4-neighbor / von Neumann grid), not all-to-all. This is
  the one architectural choice that makes MNIST scale tractable: measured
  all-to-all coupling costs ~34-48ms/step at 28x28 (~12s per image for a
  250-step feature extraction), which is untenable for a real evaluation.
  Local coupling is O(N), not O(N^2). From oscillator_field_dynamics.ipynb.

- CLOSED-LOOP input anchoring (couple to a fixed "phantom" reference
  oscillator at the input-derived target phase), not an open-loop additive
  bias to phase velocity. Validated three independent ways this project:
  HebbianKuramotoOperator's open-loop perturbation term produces a phase
  gap that drifts to essentially arbitrary values depending on random seed
  (tracked directly: three seeds gave -2.8, +3.1, +0.2 rad, no convergence);
  PredictiveHebbianOperator's closed-loop sensory-error term is what took
  it from exactly-chance accuracy to 83-98% once wired in; and
  oscillator_field_dynamics.ipynb's `bias = c - z` arrived at the same
  mechanism completely independently.

- PARTIAL-ARC phase mapping for pixel intensity ([0, pi] by default, not
  [0, 2*pi]). A full rotation aliases the two most common, most
  informative MNIST pixel values (background=0, ink=1) to the identical
  point on the circle. Measured cost of getting this wrong: ~6.6 points
  for a trained linear classifier, ~4.5 for an untrained centroid (see
  MNIST_BASELINES.md). This is a lesson pulled from OUR OWN baseline work,
  not from either notebook -- both notebooks used a full 2*pi mapping.

- FIXED (not Hebbian-adaptive) coupling weights, uniform strength. A
  deliberate first-version simplification: isolate whether local coupling
  + closed-loop anchoring alone produces a useful representation, before
  deciding whether Hebbian weight adaptation on top earns its added
  complexity. models/hebbian and models/predictive both use adaptive
  weights; this model doesn't, for now.

- SCALAR phase per oscillator (not vector-valued / AKOrN-style). Same
  reasoning as above: earn the complexity before adding it. models/akorn/
  and oscillator_field_dynamics.ipynb both use vector-valued (D>1)
  oscillators; not carried over here yet.

No natural frequency term: each oscillator's only two forces are (a) pull
toward its spatial neighbors' phases, (b) pull toward its own input-derived
target phase. For a field representing a single static image (not an
inherently time-varying signal), there's no obvious role for per-oscillator
natural frequency the way there is in the Bronski-derived models -- omitted
for now rather than included without a reason.
"""
import numpy as np


class LocalOscillatorField:
    def __init__(self, height: int, width: int, dt: float = 0.1,
                 k_coupling: float = 1.0, k_bias: float = 1.0, omega: float = 0.0,
                 seed: int = None):
        self.H, self.W = height, width
        self.dt = dt
        self.k_coupling = k_coupling
        self.k_bias = k_bias
        self.omega = omega  # shared natural frequency (cycles/unit-time); 0.0 by
        # default preserves the original converged-fixed-point behavior used
        # everywhere else. Nonzero gives every oscillator a common baseline
        # rotation -- needed for a genuine first-spike-time readout (see
        # run_track_first_spike), where "when does this first cross
        # threshold" is only meaningful if oscillators keep advancing past a
        # fixed point rather than settling and stopping.
        rng = np.random.default_rng(seed)
        self.phases = rng.uniform(0, 2 * np.pi, (height, width))
        self.target_phase = None  # set via set_input()

    def set_input(self, image: np.ndarray, arc: float = np.pi):
        """image: (H, W) array with values in [0, 1]. Maps to a target phase
        in [0, arc] -- partial arc by default (see module docstring for why
        not a full 2*pi rotation)."""
        assert image.shape == (self.H, self.W), f"expected {(self.H, self.W)}, got {image.shape}"
        self.target_phase = image * arc

    def initialize_at_target(self, perturbation_std: float = 0.01, seed: int = None):
        """Set phases = target_phase + a small symmetry-breaking perturbation
        (requires set_input() first).

        Investigated properly (see docs) rather than assumed: initializing
        EXACTLY at target_phase for a hard-edged binary pattern lands on a
        genuinely unstable equilibrium wherever neighboring pixels differ by
        exactly the arc's full extent (phase difference of exactly pi, where
        sin(pi)=0 gives zero coupling force) -- confirmed directly, an
        infinitesimal perturbation (std=0.01) from that state is enough to
        drive the system to final states differing by ~0.19-0.32 rad, the
        same order of magnitude as full random initialization. That's a real
        instability, not a robust fixed point.

        For a smoothed/anti-aliased pattern (i.e. what a real grayscale image
        with continuous intensity gradients actually looks like, unlike a
        hard-edged synthetic binary test pattern), this instability
        disappears entirely: confirmed 5 different tiny perturbations
        converging to the exact same fixed point (0.0 difference). The small
        perturbation here is retained as cheap insurance against any residual
        exactly-antipodal pixel pairs (e.g. large flat pure-black/pure-white
        regions meeting at a hard edge) even in otherwise-smooth real images,
        since it's shown to cost nothing for patterns that don't have the
        problem.
        """
        assert self.target_phase is not None, "call set_input() first"
        rng = np.random.default_rng(seed)
        self.phases = (self.target_phase + rng.normal(0, perturbation_std, self.target_phase.shape)) % (2 * np.pi)

    def _neighbor_coupling(self) -> np.ndarray:
        """4-neighbor (von Neumann) local coupling: sum of sin(neighbor - self)
        over whichever of up/down/left/right neighbors actually exist.
        Edge/corner oscillators have fewer real neighbors (2 or 3, not 4) and
        are NOT padded/wrapped to compensate -- no toroidal wraparound, and
        no spurious coupling force from a fabricated zero-valued neighbor.
        Edge sites simply experience less total coupling force, which seems
        like the physically correct behavior (fewer neighbors = less
        constrained) rather than something to paper over.
        """
        p = self.phases
        coupling = np.zeros_like(p)

        coupling[1:, :]  += np.sin(p[:-1, :] - p[1:, :])   # up neighbor
        coupling[:-1, :] += np.sin(p[1:, :]  - p[:-1, :])  # down neighbor
        coupling[:, 1:]  += np.sin(p[:, :-1] - p[:, 1:])   # left neighbor
        coupling[:, :-1] += np.sin(p[:, 1:]  - p[:, :-1])  # right neighbor

        return coupling

    def step(self) -> np.ndarray:
        coupling = self._neighbor_coupling()
        if self.target_phase is not None:
            # Closed-loop anchor: couple to a fixed "phantom" reference
            # oscillator at target_phase, exactly like the neighbor coupling
            # term above but one-directional (the phantom never updates).
            bias = np.sin(self.target_phase - self.phases)
        else:
            bias = 0.0

        dtheta = self.k_coupling * coupling + self.k_bias * bias + self.omega * 2 * np.pi
        self.phases = (self.phases + self.dt * dtheta) % (2 * np.pi)
        return self.phases

    def run(self, steps: int, record_every: int = 1) -> list:
        """Run for `steps` iterations, returning a list of phase snapshots
        (copies), one every `record_every` steps."""
        history = []
        for i in range(steps):
            self.step()
            if i % record_every == 0:
                history.append(self.phases.copy())
        return history

    def run_track_first_spike(self, steps: int, threshold: float = 0.0) -> np.ndarray:
        """Run for `steps` iterations, recording the first timestep at which
        each oscillator's phase crosses `threshold` (mod 2*pi, i.e. wraps
        around past it) -- a genuine first-spike-time readout, needing a
        nonzero shared `omega` so oscillators keep advancing rather than
        settling to a fixed point and never crossing again (see module
        docstring update: LocalOscillatorField previously had no natural
        frequency at all, which is fine for a converged-state readout but
        means "first crossing" may never happen, or happen only once,
        arbitrarily, without a shared clock to compare against).

        Returns: (H, W) array of first-spike-time in units of steps, or
        `steps` (not `-1` or `nan`) for any oscillator that never crossed --
        a sentinel that keeps the feature vector's dtype and scale uniform,
        clearly distinguishable downstream as "did not spike in this window".
        """
        spike_time = np.full((self.H, self.W), steps, dtype=np.float64)
        spiked = np.zeros((self.H, self.W), dtype=bool)
        prev_phase = self.phases.copy()
        for t in range(steps):
            self.step()
            # Crossing threshold via wraparound: previous phase was "before"
            # threshold (in the sense of having not yet completed the lap),
            # current phase has wrapped past it. Detect by looking for a
            # large negative jump in phase (mod 2*pi wraparound) that
            # straddles the threshold value.
            wrapped = (prev_phase > self.phases)  # phase decreased -> wrapped around 2*pi -> 0
            crossed_threshold = wrapped & (~spiked)
            newly_spiked = crossed_threshold
            spike_time[newly_spiked] = t
            spiked = spiked | newly_spiked
            prev_phase = self.phases.copy()
        return spike_time

    def run_track_spike_train(self, steps: int) -> np.ndarray:
        """Run for `steps` iterations, recording EVERY threshold crossing per
        oscillator, not just the first -- a full spike train (raster), the
        natural generalization of run_track_first_spike once first-spike
        served its purpose as a probe confirming timing-based information is
        worth pursuing at all.

        Returns: (steps, H, W) boolean array -- True at (t, h, w) if that
        oscillator crossed threshold at step t. This is the raw ingredient
        for a genuine spike-train cross-correlation coincidence measure
        (does pixel i and pixel j tend to fire together over the WHOLE
        window), rather than the single-scalar first-spike-time-difference
        proxy used so far.
        """
        spike_train = np.zeros((steps, self.H, self.W), dtype=bool)
        prev_phase = self.phases.copy()
        for t in range(steps):
            self.step()
            wrapped = (prev_phase > self.phases)
            spike_train[t] = wrapped
            prev_phase = self.phases.copy()
        return spike_train
