"""
Various helpers used across different experiments
"""

from typing import Callable, Tuple
import numpy as np


def generate_allocations(n_agents: int, n_goods: int) -> np.array:
    """
    Generate a list of all possible allocations (one hot encoded)

    Parameters
    ----------
    n_agents: int
    n_goods: int

    Returns
    -------
    np.array
        shape = (n_agents**n_goods, n_agents, n_goods)
        Each element is a valid allocation (n_agents, n_goods) binary matrix
        such that each column has only one value = 1 (each good assigned to exactly one agent)
    """
    allocations = []
    for i in range(n_agents**n_goods):
        str_repr = np.base_repr(i, base=n_agents).zfill(n_goods)
        allocations.append(np.eye(n_agents)[np.array([*str_repr], dtype=int)].T)
    return np.array(allocations, dtype=float)


def generate_costs(
    agent_means: np.array,
    agent_stds: np.array,
    cost_f: Callable[[np.array, np.array, np.array], np.array],
) -> Tuple[np.array, np.array]:
    """
    Generates a list of "costs" for all allocations working with gaussian valuations

    Parameters
    ----------
    agent_means : np.array
        shape = (n_agents, n_goods)
    agent_stds : np.array
        shape = (n_agents, n_goods)
    cost_f : Callable[[np.array, np.array, np.array], np.array]
        Takes as input agent_means, agent_stds, and an allocation
        Returns a cost (can be a float or a matrix)

    Returns
    -------
    Tuple[np.array, np.array]
        Returns the list of costs, and the list of possible allocations
    """
    n_agents, n_goods = agent_means.shape
    allocations = generate_allocations(n_agents, n_goods)
    costs = []
    for alloc in allocations:
        costs.append(cost_f(agent_means, agent_stds, alloc))
    return np.array(costs), allocations


def abs_means(
    agent_means: np.array, agent_stds: np.array, n_samples: int = 1000
) -> np.array:
    """
    Mean of an Abs(Gaussian) distribution

    Parameters
    ----------
    agent_means : np.array
        shape=(n_agents, n_goods)
    agent_stds : np.array
        shape=(n_agents, n_goods)
    n_samples : int, by default 1000
        Number of samples to evaluate

    Returns
    -------
    np.array
        shape=(n_agents, n_goods)
    """
    all_evals = []
    for _ in range(n_samples):
        sample = np.abs(
            agent_means + np.random.normal(0, 1, (agent_means.shape)) * agent_stds
        )
        all_evals.append(sample)
    return np.mean(all_evals, axis=0), np.std(all_evals, axis=0)


def prob_abs_gaussian_larger(mean1, var1, mean2, var2, n_samples: int = 1000):
    mean1, mean2 = np.array(mean1), np.array(mean2)
    var1, var2 = np.array(var1), np.array(var2)
    count = 0
    for _ in range(n_samples):
        s1 = mean1 + np.random.normal(0, 1, mean1.shape) * np.sqrt(var1)
        s2 = mean2 + np.random.normal(0, 1, mean2.shape) * np.sqrt(var2)
        if np.sum(s1 < s2) == 0:
            count += 1
    return count / n_samples
