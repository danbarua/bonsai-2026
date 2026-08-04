"""
Builds COMPUTE_COST_FINDINGS.md's cost-model analysis and plot
(results/compute_cost_vs_n.png). Train_readout/Infer_osc_*/Train_MLP/
Infer_MLP_ms below are transcribed directly from the measurements
already reported and verified in COMPUTE_COST_FINDINGS.md (produced by
measure_oscillator_cpu_latency.py, measure_oscillator_gpu_latency.py,
measure_mlp_cpu_latency.py, and the stage-3/cuml.accel CV-timing
numbers already in FINDINGS.md/CUML_ACCEL_FINDINGS.md) -- this script
does not re-derive them from raw artifacts itself, only builds the cost
model and plot from the already-verified values.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- All numbers below are from this session's own measurements/existing FINDINGS ---

Train_MLP = {"MLP_H13": 6.3, "MLP_H128": 26.4}  # seconds, CPU sklearn
Infer_MLP_ms = {"MLP_H13": 0.0525, "MLP_H128": 0.0534}  # ms/image, CPU sklearn

Train_readout = {"T": 361.2, "lattice": 308.0, "rewired": 379.3, "curr_random": 262.3}  # seconds
Infer_osc_cpu_ms = {"T": 322.33, "lattice": 321.79, "rewired": 281.98, "curr_random": 278.16}
Infer_osc_gpu_ms = {"T": 29.61, "lattice": 29.30, "rewired": 29.67, "curr_random": 29.71}

N_values = np.logspace(1, 8, 500)  # N=10 to N=1e8

def total_cost(train_s, infer_ms, N):
    return train_s + N * (infer_ms / 1000.0)

def find_breakeven(train_a, infer_a_ms, train_b, infer_b_ms):
    """Solve train_a + N*infer_a = train_b + N*infer_b for N (seconds/ms consistent units)."""
    infer_a_s = infer_a_ms / 1000.0
    infer_b_s = infer_b_ms / 1000.0
    denom = infer_a_s - infer_b_s
    if abs(denom) < 1e-15:
        return None  # parallel lines, no crossing (or identical)
    N = (train_b - train_a) / denom
    return N

print("="*70)
print("BREAK-EVEN ANALYSIS: oscillator (GPU) vs. each MLP baseline")
print("="*70)
for osc_name in Train_readout:
    for mlp_name in Train_MLP:
        N_be = find_breakeven(Train_readout[osc_name], Infer_osc_gpu_ms[osc_name],
                               Train_MLP[mlp_name], Infer_MLP_ms[mlp_name])
        print(f"  {osc_name} (GPU) vs {mlp_name}: break-even N = {N_be}")

print("\n" + "="*70)
print("REPRESENTATIVE N VALUES: Total cost (seconds), oscillator=evolved_T (GPU) vs MLP_H128")
print("="*70)
for N in [1, 10, 1000, 1_000_000, 100_000_000]:
    osc_cost = total_cost(Train_readout["T"], Infer_osc_gpu_ms["T"], N)
    mlp_cost = total_cost(Train_MLP["MLP_H128"], Infer_MLP_ms["MLP_H128"], N)
    ratio = osc_cost / mlp_cost
    print(f"  N={N:>12,}: oscillator(T,GPU)={osc_cost:>14,.2f}s, MLP_H128={mlp_cost:>10,.2f}s, "
          f"ratio={ratio:>8.1f}x")

# --- Rough FLOPs estimate ---
print("\n" + "="*70)
print("ROUGH FLOPS ESTIMATE (order-of-magnitude, hardware-independent cross-check)")
print("="*70)
# MLP forward pass: 784*H + H*10 multiply-adds (x2 for mult+add), ignoring bias/activation (small)
for H, name in [(13, "MLP_H13"), (128, "MLP_H128")]:
    flops = 2 * (784 * H + H * 10)
    print(f"  {name}: ~{flops:,} FLOPs/image (forward pass matrix multiplies)")

# Oscillator encode: 150 steps x 784 nodes x ~5 trig-op-equivalents (4 neighbor + 1 bias sin)
encode_flops = 150 * 784 * 5
print(f"  Oscillator encode: ~{encode_flops:,} trig-op-equivalents/image (150 steps x 784 nodes x 5)")

# Oscillator evolve: each RHS eval is O(n^2) (n=505): diff + sin + weighted-sum ~3 ops/entry
n = 505
rhs_eval_flops = n * n * 3
for n_rhs_evals in [10, 50, 100]:
    total_evolve = rhs_eval_flops * n_rhs_evals
    print(f"  Oscillator evolve @ ~{n_rhs_evals} RHS evals: ~{total_evolve:,} FLOPs/image "
          f"({rhs_eval_flops:,}/RHS-eval x {n_rhs_evals})")

# --- Plot ---
fig, ax = plt.subplots(figsize=(9, 6))
colors = {"T": "tab:blue", "lattice": "tab:orange", "rewired": "tab:green", "curr_random": "tab:red"}
for osc_name in Train_readout:
    costs = total_cost(Train_readout[osc_name], Infer_osc_gpu_ms[osc_name], N_values)
    ax.plot(N_values, costs, label=f"oscillator: evolved_{osc_name} (GPU)", color=colors[osc_name])

for mlp_name, ls in [("MLP_H13", "--"), ("MLP_H128", ":")]:
    costs = total_cost(Train_MLP[mlp_name], Infer_MLP_ms[mlp_name], N_values)
    ax.plot(N_values, costs, label=mlp_name, color="black", linestyle=ls)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("N (number of images classified)")
ax.set_ylabel("Total wall-clock cost (seconds)")
ax.set_title("Total compute cost vs. deployment scale\n(oscillator: GPU evolution; MLP: CPU sklearn)")
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(_THIS_DIR, "results", "compute_cost_vs_n.png")
plt.savefig(out_path, dpi=150)
print(f"\nSaved plot to {out_path}")
