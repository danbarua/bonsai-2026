"""
Pure-NumPy loader for MNIST's IDX file format (no sklearn/torch dependency).

Format reference (http://yann.lecun.com/exdb/mnist/, mirrored at
https://ossci-datasets.s3.amazonaws.com/mnist/):
  IDX images: magic(4 bytes, big-endian uint32) = 2051, then n_images(4),
              n_rows(4), n_cols(4), then n_images*n_rows*n_cols bytes (uint8).
  IDX labels: magic(4 bytes) = 2049, then n_labels(4), then n_labels bytes (uint8).

Files are gzip-compressed (.gz) as distributed.
"""
import gzip
import struct
import numpy as np


def load_idx_images(path: str) -> np.ndarray:
    """Load an IDX3 (images) file, gzipped or not. Returns (N, H, W) uint8 array."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        magic, n_images, n_rows, n_cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Bad magic number for IDX images file: {magic} (expected 2051)")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n_images, n_rows, n_cols)


def load_idx_labels(path: str) -> np.ndarray:
    """Load an IDX1 (labels) file, gzipped or not. Returns (N,) uint8 array."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        magic, n_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Bad magic number for IDX labels file: {magic} (expected 2049)")
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_mnist(data_dir: str = "."):
    """
    Load standard MNIST train/test split from IDX files in data_dir.
    Expects: train-images-idx3-ubyte.gz, train-labels-idx1-ubyte.gz,
             t10k-images-idx3-ubyte.gz, t10k-labels-idx1-ubyte.gz

    Returns: (X_train, y_train, X_test, y_test)
      X_*: (N, 28, 28) uint8 arrays (raw pixel values, 0-255)
      y_*: (N,) uint8 arrays (digit labels, 0-9)
    """
    import os
    X_train = load_idx_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
    y_train = load_idx_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    X_test = load_idx_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
    y_test = load_idx_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"Train: {X_train.shape} images, {y_train.shape} labels")
    print(f"Test:  {X_test.shape} images, {y_test.shape} labels")
    print(f"Label distribution (train): {np.bincount(y_train)}")
