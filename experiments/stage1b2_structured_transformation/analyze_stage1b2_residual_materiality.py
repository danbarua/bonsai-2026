"""
Residual-materiality check, per review: the permutation test establishes
that the nonlinear residual's NORMALIZED spatial geometry (q_residual) is
reproducibly associated with input identity. It does not, by itself,
establish that the residual is materially large -- a small residual can
still produce a sharply organized normalized pattern. This reports the
absolute size of the residual and how many trials clear the numerical-
validity and nonlinear-departure thresholds, broken down by amplitude
and perturbation time.
"""
import numpy as np
import pickle
from stage1b2_core import T_P_VALUES, AMPLITUDES, SIGNS, N_REPLICAS, Q_NORM_THRESHOLD, E_MIN, get_degree_stratified_nodes


def summarize(results, nodes):
    node_labels = list(nodes.keys())

    print(f'{"="*78}\nRESIDUAL MATERIALITY, BY AMPLITUDE\n{"="*78}')
    print(f'{"amplitude":<12}{"median||z||":<14}{"IQR||z||":<20}{"median E":<12}{"n>=Qthr":<10}{"n>=Emin":<10}{"undefined":<10}{"total"}')
    for amp in AMPLITUDES:
        rows = [v for k, v in results.items() if k[4] == amp]
        norms = np.array([r['event_aligned_residual_norm'] for r in rows if r['event_aligned_residual_norm'] is not None])
        Es = np.array([r['E_at_tau_star'] for r in rows])
        n_valid_q = np.sum([r['event_aligned_q_residual'] is not None for r in rows])
        n_valid_e = np.sum([r['event_aligned_valid'] for r in rows])
        n_undefined = np.sum([r['event_aligned_q_residual'] is None for r in rows])
        total = len(rows)
        if len(norms) > 0:
            med_norm = np.median(norms)
            iqr_norm = (np.percentile(norms, 25), np.percentile(norms, 75))
        else:
            med_norm, iqr_norm = float('nan'), (float('nan'), float('nan'))
        med_e = np.median(Es)
        print(f'{amp:<12}{med_norm:<14.5f}[{iqr_norm[0]:.5f},{iqr_norm[1]:.5f}]{"":<2}{med_e:<12.5f}'
              f'{n_valid_q:<10}{n_valid_e:<10}{n_undefined:<10}{total}')

    print(f'\n{"="*78}\nRESIDUAL MATERIALITY, BY PERTURBATION TIME\n{"="*78}')
    print(f'{"t_p":<12}{"median||z||":<14}{"IQR||z||":<20}{"median E":<12}{"n>=Qthr":<10}{"n>=Emin":<10}{"undefined":<10}{"total"}')
    for t_p in T_P_VALUES:
        rows = [v for k, v in results.items() if k[0] == t_p]
        norms = np.array([r['event_aligned_residual_norm'] for r in rows if r['event_aligned_residual_norm'] is not None])
        Es = np.array([r['E_at_tau_star'] for r in rows])
        n_valid_q = np.sum([r['event_aligned_q_residual'] is not None for r in rows])
        n_valid_e = np.sum([r['event_aligned_valid'] for r in rows])
        n_undefined = np.sum([r['event_aligned_q_residual'] is None for r in rows])
        total = len(rows)
        if len(norms) > 0:
            med_norm = np.median(norms)
            iqr_norm = (np.percentile(norms, 25), np.percentile(norms, 75))
        else:
            med_norm, iqr_norm = float('nan'), (float('nan'), float('nan'))
        med_e = np.median(Es)
        print(f'{t_p:<12}{med_norm:<14.5f}[{iqr_norm[0]:.5f},{iqr_norm[1]:.5f}]{"":<2}{med_e:<12.5f}'
              f'{n_valid_q:<10}{n_valid_e:<10}{n_undefined:<10}{total}')

    print(f'\n{"="*78}\nOVERALL\n{"="*78}')
    all_norms = np.array([r['event_aligned_residual_norm'] for r in results.values() if r['event_aligned_residual_norm'] is not None])
    all_Es = np.array([r['E_at_tau_star'] for r in results.values()])
    n_valid_q = np.sum([r['event_aligned_q_residual'] is not None for r in results.values()])
    n_valid_e = np.sum([r['event_aligned_valid'] for r in results.values()])
    n_undefined = np.sum([r['event_aligned_q_residual'] is None for r in results.values()])
    total = len(results)
    print(f'median ||z_eps(tau*)|| = {np.median(all_norms):.5f}, '
          f'IQR = [{np.percentile(all_norms,25):.5f}, {np.percentile(all_norms,75):.5f}]')
    print(f'median E_eps(tau*) = {np.median(all_Es):.5f}')
    print(f'trials with defined q_residual (||z|| >= Q_NORM_THRESHOLD={Q_NORM_THRESHOLD}): '
          f'{n_valid_q}/{total} ({100*n_valid_q/total:.1f}%)')
    print(f'trials with event_aligned_valid (E(tau*) >= E_MIN={E_MIN}): '
          f'{n_valid_e}/{total} ({100*n_valid_e/total:.1f}%)')
    print(f'trials with undefined residual map (excluded from residual analysis): '
          f'{n_undefined}/{total} ({100*n_undefined/total:.1f}%)')


if __name__ == '__main__':
    with open('results/stage1b2_results.pkl', 'rb') as f:
        results = pickle.load(f)
    with open('results/class0_constructions.pkl', 'rb') as f:
        data = pickle.load(f)[0]
    W_matrix = data['constructions']['T']
    nodes = get_degree_stratified_nodes(W_matrix)
    print(f'Loaded {len(results)} trials (expect 432)')
    summarize(results, nodes)
