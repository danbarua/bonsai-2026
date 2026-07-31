# Sharpened Steps 2-3 — HopfCochlea and Driven-Oscillator Verification

## Step 2: `HopfCochlea` — no coupling in v1, and get the scaling exactly right

Write this as a **new class**, not an inheritance/refactor of
`ComplexLocalOscillatorField`. That class's `step()` has neighbor-coupling
built into its core update — reusing it risks coupling sneaking into a
model that shouldn't have any yet. Borrow the verified intrinsic-Hopf math
only; write the bank structure fresh.

**Exact governing equation, per oscillator `i`, NO coupling term:**

```
dz_i/dt = (mu_i + i)*omega_i*z_i - omega_i*|z_i|^2*z_i - omega_i*F(t)
```

- `z_i`: complex state of oscillator `i`.
- `mu_i`: distance from the Hopf bifurcation. `mu_i = 0` is the critical
  (interesting) tuning; `mu_i < 0` gives a damped, stable, off-critical
  oscillator (needed as a control in Step 3, not optional).
- `omega_i`: oscillator `i`'s characteristic frequency.
- `F(t)`: real-valued, shared across every oscillator in the bank -- the
  audio input signal.

**Important, easy-to-miss detail**: all three terms are scaled by `omega_i`,
including the `|z_i|^2*z_i` and `F(t)` terms. This differs from
`ComplexLocalOscillatorField`'s intrinsic term
(`z*(lambda + i*omega - |z|^2)`, unscaled) -- that was fine for the MNIST
work's shared, arbitrary time-unit setting, but here `omega_i` varies
across the bank (different oscillators, different characteristic
frequencies) and the scaling is part of the actual cited model, not a
stylistic choice. Implement the scaled version exactly as written above,
not the MNIST version's unscaled one.

**Bank structure**: N oscillators, `omega_i` log-spaced across an audible
range (mirrors real cochlear tonotopic organization -- e.g.
`np.logspace(log10(100), log10(4000), N)` as a starting range, adjust once
basic behavior is confirmed). Every oscillator receives the identical
`F(t)`. No cross-oscillator coupling term anywhere in this step.

## Step 3: verification tests -- specific, falsifiable targets, not "looks nonlinear"

All three tests need steady-state output, not transient: discard an
initial settling period (a few periods of the drive is not enough --
confirm settling the same way the MNIST work did, by checking the output
stops changing step-to-step before analysis) and simulate long enough for
clean FFT frequency resolution (`frequency_resolution = 1/T_total` -- if
distinguishing components a few Hz apart, `T_total` needs to be at least
several *seconds* of simulated time, not a handful of cycles).

### Test A: single-tone harmonic generation

- Drive one oscillator (tuned on-resonance, `omega_i ≈ 2*pi*f`) with
  `F(t) = A*sin(2*pi*f*t)`.
- FFT the steady-state output. Assert: a dominant peak at `f`, AND
  measurable peaks at `2f`, `3f` above the noise floor -- the actual,
  named target, not "spectrum looks broadened."

### Test B: amplitude compression, WITH the off-critical control

This is the one place a control is not optional -- the claim being tested
is specifically that criticality (`mu=0`) causes compression, not that
"Hopf oscillators are nonlinear in general."

- Sweep input amplitude `A` over several orders of magnitude (log-spaced,
  e.g. `np.logspace(-3, 0, 15)`), small-to-moderate range (the sub-linear
  law is a small-signal asymptotic prediction -- expect it to break down at
  large `A`, and that breakdown is fine, not a failure).
- For each `A`, measure steady-state output amplitude (`|z|` at resonance,
  or peak FFT magnitude at `f`).
- **At `mu=0`**: fit `log(output) vs log(A)` by linear regression. Assert
  the fitted slope is close to `1/3` (e.g. within +/-0.1) -- the specific,
  citable exponent from the Hopf-cochlea literature, not just "sub-linear."
- **At `mu<0` (e.g. `mu=-1`), same sweep, same fit**: assert the slope is
  close to `1.0` (linear response) -- confirming compression specifically
  requires criticality. If both conditions give the same slope, something
  is wrong with the implementation or the test, not confirmation of the
  effect.

### Test C: two-tone combination products, with a linear control

- Drive with `F(t) = A*sin(2*pi*f1*t) + A*sin(2*pi*f2*t)`, `f1` and `f2`
  close together, both near an oscillator's `omega_i`.
- FFT the steady-state output. Assert measurable peaks specifically at
  `2*f1 - f2` and `2*f2 - f1` (the named cubic distortion products from
  the cochlear otoacoustic-emission literature) above the noise floor --
  not "new peaks appear somewhere."
- **Control**: repeat with a plain damped linear oscillator (drop the
  `-|z|^2*z` term entirely, or set `mu` very negative and check the
  nonlinear term's contribution is negligible) and assert these specific
  peaks are absent or far weaker -- confirming the products come from the
  Hopf nonlinearity, not simulation/FFT artifacts.

## Explicitly deferred, don't pull forward

Real `.wav` files (Step 4) aren't needed for any of the above -- synthetic
`numpy` sine tones give exact, controlled frequencies and amplitudes to
check against exact theoretical predictions, which is what these tests
need. Bring in real audio once A/B/C pass cleanly, not before.
