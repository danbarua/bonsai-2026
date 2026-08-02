import pickle
import numpy as np

def main():
    with open('../stage1b2_structured_transformation/results/class0_constructions.pkl', 'rb') as f:
        data = pickle.load(f)

    T = data[0]['constructions']['T']
    n_active = data[0]['n_active']
    print(f"T shape: {T.shape}, dtype: {T.dtype}")
    print(f"n_active: {n_active}")
    print(f"nonzero edges: {np.count_nonzero(T)}")
    print(f"symmetric: {np.allclose(T, T.T)}")
    np.save('class0_T.npy', T.astype(np.float64))
    print("Saved class0_T.npy")

if (__name__ == '__main__'):
    main()