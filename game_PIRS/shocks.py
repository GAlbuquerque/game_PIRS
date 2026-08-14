"""Random macroeconomic shocks used by the event and motion pipeline."""

import numpy as np


def generate_shocks(correlation_matrix, standard_deviations):
    """Draw correlated inflation, demand, natural-rate, and equilibrium-rate shocks."""
    standard_deviations = np.asarray(standard_deviations, dtype=float)
    covariance = (
        np.diag(standard_deviations)
        @ correlation_matrix
        @ np.diag(standard_deviations)
    )
    return np.random.multivariate_normal(np.zeros(len(standard_deviations)), covariance)
