"""
Topology-as-representation: classify by how well a test image's OWN
per-image developmental topology matches each class's reference topology
(the specific (i,j) connections that survived pruning for that class) --
not by running dynamics through a borrowed topology (confirmed broken:
halved accuracy, since forcing class-1's connections onto a class-0 image
corrupts its natural dynamics rather than enriching them).

Each class's reference topology is a sparse mask of "diagnostic positions"
-- pairs that reliably co-activate across that class's population. The
test image's own per-image topology (built the identical way, but from
just one image) is scored by how well it agrees (same sign, comparable
magnitude) at exactly those positions. No borrowed structure ever gets
wired into any dynamics -- topology stays purely a comparison object.
"""
import numpy as np
from developmental_pruning import get_local_converged_phases

H, W = 28, 28


def per_image_topology(image, steps=150):
    """This image's own full pairwise cos(theta_i - theta_j) matrix --
    the per-image analog of population_developmental_stat, from just one
    image's dynamics."""
    phases = get_local_converged_phases(image, steps=steps)
    p = phases.flatten()
    diff = p[:, None] - p[None, :]
    return np.cos(diff)


def topology_match_score(per_image_stat, class_topology):
    """How well does this image's own topology agree with the class's
    diagnostic (surviving-connection) positions -- same sign, weighted by
    magnitude, only at those specific sparse positions."""
    mask = np.abs(class_topology) > 0
    if mask.sum() == 0:
        return 0.0
    return np.mean(per_image_stat[mask] * np.sign(class_topology[mask]))


def classify_by_topology_match(image, class_topologies, steps=150):
    per_image_stat = per_image_topology(image, steps=steps)
    scores = {c: topology_match_score(per_image_stat, topo) for c, topo in class_topologies.items()}
    predicted = max(scores, key=scores.get)
    return predicted, scores


def compute_class_baselines(calibration_images, class_topologies):
    """Mean/std of each class's raw match score against a broad, mixed
    (not class-specific) calibration population -- the 'how well does this
    topology match ANY image' null expectation, used to z-score raw scores
    before comparing across classes. Computes per_image_topology ONCE per
    calibration image, reusing it against all classes (cheap, since
    topology_match_score is just a masked mean)."""
    scores_by_class = {c: [] for c in class_topologies}
    for image in calibration_images:
        stat = per_image_topology(image)
        for c, topo in class_topologies.items():
            scores_by_class[c].append(topology_match_score(stat, topo))
    return {c: (np.mean(scores), np.std(scores)) for c, scores in scores_by_class.items()}


def classify_by_normalized_topology_match(image, class_topologies, baselines, steps=150):
    per_image_stat = per_image_topology(image, steps=steps)
    z_scores = {}
    for c, topo in class_topologies.items():
        raw_score = topology_match_score(per_image_stat, topo)
        mean_b, std_b = baselines[c]
        z_scores[c] = (raw_score - mean_b) / (std_b + 1e-12)
    predicted = max(z_scores, key=z_scores.get)
    return predicted, z_scores


def topology_match_score_v2(per_image_stat, class_topology, mode="cosine"):
    """Alternative match-score formulations, suggested in code review --
    testing directly rather than assuming any of these beat the original
    masked-mean-times-sign version."""
    mask = np.abs(class_topology) > 0
    if mask.sum() == 0:
        return 0.0
    ref_vals = class_topology[mask]
    test_vals = per_image_stat[mask]

    if mode == "simple":
        return np.mean(test_vals * np.sign(ref_vals))
    elif mode == "weighted":
        return np.average(test_vals * np.sign(ref_vals), weights=np.abs(ref_vals))
    elif mode == "cosine":
        return np.dot(test_vals, ref_vals) / (np.linalg.norm(test_vals) * np.linalg.norm(ref_vals) + 1e-12)
    else:
        raise ValueError(f"unknown mode {mode}")
