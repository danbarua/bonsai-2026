import pickle
import numpy as np

def main():
    with open('../stage1b2_structured_transformation/results/class0_constructions.pkl', 'rb') as f:
        data = pickle.load(f)

    lattice = data[0]['constructions']['lattice']
    print(f"lattice shape: {lattice.shape}, total weight: {lattice.sum()}")
    np.save('class0_lattice.npy', lattice.astype(np.float64))
    print("Saved lattice to class0_lattice.npy")

if (__name__ == '__main__'):
    main()