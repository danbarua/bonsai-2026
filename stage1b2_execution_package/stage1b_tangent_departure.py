"""
Computes E_eps(t) (amplitude departure from tangent prediction) and
C_eps(t) (directional cosine similarity) for a given (graph, IC, node,
epsilon) trial, alongside the finite-amplitude trajectory -- disambiguates
tangent-linear transient growth from genuine nonlinear effects, which
peak-amplification alone conflates.
"""
import numpy as np
from scipy.integrate import solve_ivp
from graph_oscillator_field import rotation_projector, force_jacobian

def finite_and_tangent_comparison(W, theta0, perturb_node, epsilon, t_span, k_coupling=1.0,
                                    rtol=1e-6, atol=1e-8, max_step=0.05, t_eval=None):
    n = len(theta0)
    P = rotation_projector(n)

    delta0 = np.zeros(n); delta0[perturb_node] = 1.0
    delta0 = P @ delta0; delta0 = delta0 / np.linalg.norm(delta0)

    # Joint baseline + tangent
    y0_tan = np.concatenate([theta0, delta0])
    def rhs_tan(t, y):
        theta = y[:n]; delta = y[n:]
        diff = theta[None,:]-theta[:,None]
        dtheta = k_coupling*np.sum(W*np.sin(diff),axis=1)
        DF = force_jacobian(W, theta, k_coupling=k_coupling)
        return np.concatenate([dtheta, DF@delta])
    sol_tan = solve_ivp(rhs_tan, t_span, y0_tan, method='RK45', rtol=rtol, atol=atol,
                        t_eval=t_eval, max_step=max_step)
    theta_base_t = sol_tan.y[:n,:]
    delta_t = sol_tan.y[n:,:]

    # Finite perturbed
    def rhs(t, theta):
        diff = theta[None,:]-theta[:,None]
        return k_coupling*np.sum(W*np.sin(diff),axis=1)
    theta0_pert = theta0 + epsilon*delta0
    sol_pert = solve_ivp(rhs, t_span, theta0_pert, method='RK45', rtol=rtol, atol=atol,
                        t_eval=t_eval, max_step=max_step)
    theta_pert_t = sol_pert.y

    eta = 1e-10
    E_list, C_list, S_list = [], [], []
    D0 = None
    for i in range(theta_pert_t.shape[1]):
        shift = np.angle(np.mean(np.exp(1j*(theta_pert_t[:,i]-theta_base_t[:,i]))))
        actual_disp = P @ np.angle(np.exp(1j*(theta_pert_t[:,i]-theta_base_t[:,i]-shift)))
        predicted_disp = epsilon * (P @ delta_t[:,i])
        E = np.linalg.norm(actual_disp-predicted_disp) / (abs(epsilon)*np.linalg.norm(P@delta_t[:,i])+eta)
        denom = np.linalg.norm(actual_disp)*np.linalg.norm(predicted_disp)
        C = np.dot(actual_disp,predicted_disp)/denom if denom>eta else np.nan
        E_list.append(E); C_list.append(C)
        D_raw = np.sum(actual_disp**2)
        if D0 is None: D0 = D_raw
        S_list.append(D_raw/D0 if D0>1e-15 else np.nan)

    return np.array(E_list), np.array(C_list), np.array(S_list)
