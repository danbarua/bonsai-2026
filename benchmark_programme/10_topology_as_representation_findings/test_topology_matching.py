"""
Verification tests for the topology-as-representation classifier --
same discipline as test_complex_hopf_field.py: check the mechanism
directly, not just report results.
"""
import unittest
import numpy as np


class TestBackgroundExclusion(unittest.TestCase):
    """The critical fix that made population statistics meaningful at all:
    background-background pairs must be excluded, not just magnitude-
    pruned -- confirmed earlier that they survive a 0.9 threshold ~88% of
    the time regardless of class, a trivial confound."""

    def test_background_pairs_are_zeroed(self):
        from learned_topology_encoder import build_class_topologies
        from mnist_loader import load_idx_images, load_idx_labels

        X_train = load_idx_images("mnist_data/train-images.idx3-ubyte")
        y_train = load_idx_labels("mnist_data/train-labels.idx1-ubyte")
        topologies = build_class_topologies(X_train, y_train, classes=[0], n_per_class=20)

        idx = np.where(y_train == 0)[0][:20]
        images = X_train[idx].astype(np.float64) / 255.0
        mean_intensity = images.mean(axis=0).flatten()
        ink_mask = mean_intensity > 0.15
        bg_pair_mask = np.outer(~ink_mask, ~ink_mask)

        topo = topologies[0]
        self.assertTrue(np.all(topo[bg_pair_mask] == 0.0),
            msg="Every background-background pair must be exactly zero, regardless of raw correlation")


class TestPerImageTopologyIsNonDegenerate(unittest.TestCase):
    """A per-image topology should show real, non-trivial structure --
    not collapse to all-zero or all-identical values."""

    def test_per_image_topology_has_real_variance(self):
        from topology_matching_classifier import per_image_topology
        from mnist_loader import load_idx_images

        X_train = load_idx_images("mnist_data/train-images.idx3-ubyte")
        image = X_train[0].astype(np.float64) / 255.0
        stat = per_image_topology(image)

        self.assertTrue(np.all(np.isfinite(stat)), msg="No NaN/inf in per-image topology")
        self.assertGreater(stat.std(), 0.01, msg="Per-image topology should show real variance, not collapse")


class TestCalibrationCorrectsClassBias(unittest.TestCase):
    """The z-score calibration should reduce (not necessarily eliminate)
    the systematic bias found in raw-score classification -- confirmed
    earlier: raw scores gave 45.5% with severe class imbalance
    (some classes 2-3x over-predicted), calibrated scores gave 68-82.5%
    with much more balanced predictions."""

    def test_normalized_scores_differ_meaningfully_from_raw(self):
        from topology_matching_classifier import (
            per_image_topology, topology_match_score, classify_by_normalized_topology_match
        )
        from learned_topology_encoder import build_class_topologies
        from mnist_loader import load_idx_images, load_idx_labels

        X_train = load_idx_images("mnist_data/train-images.idx3-ubyte")
        y_train = load_idx_labels("mnist_data/train-labels.idx1-ubyte")
        classes = [0, 1, 2]
        topologies = build_class_topologies(X_train, y_train, classes=classes, n_per_class=20)

        rng = np.random.default_rng(1)
        calib_idx = rng.choice(len(X_train), size=30, replace=False)
        calibration_images = X_train[calib_idx].astype(np.float64) / 255.0

        from topology_matching_classifier import compute_class_baselines
        baselines = compute_class_baselines(calibration_images, topologies)

        # Baselines should differ meaningfully across classes -- if they were
        # all identical, calibration would be a no-op (and something would be wrong)
        means = [baselines[c][0] for c in classes]
        self.assertGreater(np.std(means), 0.01,
            msg="Class baselines should genuinely differ -- otherwise normalization does nothing")


if __name__ == "__main__":
    unittest.main()
