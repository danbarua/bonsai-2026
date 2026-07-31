"""
Single fixed random seed used across every stochastic operation in this
rebuild: dataset stratified sampling, calibration image selection, random
projection controls, shuffled-spectral controls, and any other place a
seed was previously chosen ad hoc (555, 999, 2024, 0, 1, 2, etc. across
different parts of this project's history). Using one constant throughout
makes the whole pipeline reproducible from a single parameter and may
surface issues that varying seeds could have masked.
"""
SEED = 42
