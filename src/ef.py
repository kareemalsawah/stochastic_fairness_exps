"""
Different functions related to EF (envy-free) definitions
"""

import numpy as np
import scipy.stats as stats

from utils import prob_abs_gaussian_larger, abs_means


def is_ef1(vals: np.array, alloc: np.array) -> bool:
    """
    Testing is a deterministic valuation and allocation are EF1

    Parameters
    ----------
    vals : np.array
        Deterministic valuations, shape=(n_agents, n_goods)
    alloc : np.array
        An allocation, binary matrix with shape=(n_agents, n_goods)

    Returns
    -------
    bool
        True if this pair is EF1, False otherwise
    """
    n_agents, _ = vals.shape

    for i in range(n_agents):
        vi_Ai = np.sum(vals[i] * alloc[i])
        for j in range(n_agents):
            Aj = alloc[j]
            if np.sum(Aj) > 0:  # If Aj is not empty
                # Find the best good to remove
                best_good_idx = np.argmax(vals[i] * Aj)
                newAj = np.copy(Aj)
                newAj[best_good_idx] = 0
                vi_Aj = np.sum(vals[i] * newAj)
                if (
                    vi_Ai < vi_Aj
                ):  # agent i still envies agent j after removing the most valuable good in Aj
                    return False
    return True


def prob_ef1_cost(
    agent_means: np.array, agent_stds: np.array, alloc: np.array, n_samples: int = 1000
) -> float:
    """
    Sample estimate of the probability of Abs(Gaussian) valuations to be EF1

    Parameters
    ----------
    agent_means : np.array
        shape=(n_agents, n_goods)
    agent_stds : np.array
        shape=(n_agents, n_goods)
    alloc : np.array
        shape=(n_agents, n_goods)
    n_samples : int, by default 1000
        Number of sample valuations to generate

    Returns
    -------
    float
        Prob of a random valuation being EF1 for the given allocation
    """
    count_ef = 0
    for _ in range(n_samples):
        sample = np.abs(
            agent_means + np.random.normal(0, 1, (agent_means.shape)) * agent_stds
        )
        if is_ef1(sample, alloc):
            count_ef += 1
    return count_ef / n_samples


def alpha_ef1_cost(
    agent_means: np.array, agent_stds: np.array, alloc: np.array
) -> float:
    """
    For a given allocation, find the largest alpha such that
    E[vi(Ai)] >= min over g (E[vi(Aj\{g})]) + alpha, for all i,j

    Parameters
    ----------
    agent_means : np.array
        shape=(n_agents, n_goods)
    agent_stds: np.array
        Not Used, added for compatibility with generate_costs function
    alloc : np.array
        shape=(n_agents, n_goods)

    Returns
    -------
    float
        The largest alpha possible such that for all i,j
            E[vi(Ai)] >= min over g (E[vi(Aj\{g})]) + alpha
        If alpha < 0, then this allocation is not EF1 (in the deterministic sense)
    """
    means = abs_means(agent_means, agent_stds)
    n_agents, _ = means.shape
    min_alpha = np.inf

    for i in range(n_agents):
        vi_Ai = np.sum(means[i] * alloc[i])
        for j in range(n_agents):
            Aj = alloc[j]
            if np.sum(Aj) > 0:  # Aj is not empty
                best_good_idx = np.argmax(means[i] * Aj)
                newAj = np.copy(Aj)
                newAj[best_good_idx] = 0
                vi_Aj = np.sum(means[i] * newAj)
                alpha = vi_Ai - vi_Aj
                if alpha < min_alpha:
                    min_alpha = alpha
    return min_alpha


def em1_gaussian_cost_matrix(
    agent_means: np.array, agent_stds: np.array, alloc: np.array
):
    """
    EM1 cost matrix

    Parameters
    ----------
    agent_means : np.array
        shape=(n_agents, n_goods)
    agent_stds : np.array
        shape=(n_agents, n_goods)
    alloc : np.array
        shape=(n_agents, n_goods)

    Returns
    -------
    _type_
        _description_
    """
    n_agents, n_goods = agent_means.shape
    agent_vars = agent_stds * agent_stds
    means = np.ones((n_agents, n_agents, n_goods))
    variances = np.ones((n_agents, n_agents, n_goods))
    is_valid = np.ones((n_agents, n_agents, n_goods))

    for i in range(n_agents):
        for j in range(n_agents):
            for k in range(n_goods):
                # get V_i(A_i) - V_i(A_j-good_k), if good_k in A_j
                # otherwise, sets validity matrix to zero (if good k not in A_j and A_j is not empty)
                A_j = np.copy(alloc[j])
                if A_j[k] == 0 and np.sum(A_j) > 0:
                    is_valid[i][j][k] = 0
                else:
                    A_j[k] = 0
                    new_alloc = alloc[i] - A_j
                    means[i][j][k] = np.sum(agent_means[i] * new_alloc)
                    variances[i][j][k] = np.sum(agent_vars[i] * np.abs(new_alloc))

    prob_is_positive = 1 - stats.norm.cdf(0, means, np.sqrt(variances) + 1e-3)
    prob_is_positive *= is_valid
    max_probs = np.max(prob_is_positive, axis=2)
    return max_probs


def em1_abs_gaussian_cost_matrix(
    agent_means: np.array, agent_stds: np.array, alloc: np.array
):
    """
    EM1 cost matrix

    Parameters
    ----------
    agent_means : np.array
        shape=(n_agents, n_goods)
    agent_stds : np.array
        shape=(n_agents, n_goods)
    alloc : np.array
        shape=(n_agents, n_goods)

    Returns
    -------
    _type_
        _description_
    """
    n_agents, n_goods = agent_means.shape
    agent_vars = agent_stds * agent_stds
    max_probs = np.ones((n_agents, n_goods))

    for i in range(n_agents):
        mean1 = np.sum(agent_means[i] * alloc[i])
        var1 = np.sum(agent_vars[i] * alloc[i])
        for j in range(n_agents):
            probs = []
            if np.sum(alloc[j]) > 0:
                for k in range(n_goods):
                    A_j = np.copy(alloc[j])
                    if A_j[k] == 0:
                        pass
                    else:
                        A_j[k] = 0
                        mean2 = np.sum(agent_means[i] * A_j)
                        var2 = np.sum(agent_vars[i] * A_j)
                        probs.append(prob_abs_gaussian_larger(mean1, var1, mean2, var2))
                max_probs[i][j] = max(probs)

    return np.array(max_probs)
