from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up, Product, Sum
from spn.structure.leaves.parametric.Parametric import Bernoulli

import numpy as np


def create_naive_fact(data, scope, alpha):
    """
    It returns a naive factorization of the data.
    Laplace's correction is not needed, but if not used may cause underflow.
    """

    probs = (np.sum(data, axis=0) + alpha) / (data.shape[0] + 2 * alpha)

    naive_fact = Product(children=[Bernoulli(p=probs[k], scope=[scope[k]]) for k in range(len(scope))])

    assign_ids(naive_fact)
    rebuild_scopes_bottom_up(naive_fact)

    return naive_fact


def create_conj(data, scope, alpha):

    conj = Product(
        children=[Bernoulli(scope=[scope[k]], p=(data[0][k] * data.shape[0] + alpha) / (data.shape[0] + 2 * alpha))
                  for k in range(len(scope))])

    assign_ids(conj)
    rebuild_scopes_bottom_up(conj)

    return conj


def create_disj(data, scope, assignments, alpha):

    unq_data, counts = np.unique(data, axis=0, return_counts=True)
    probs = np.zeros(assignments.shape[0])
    for i in range(assignments.shape[0]):
        index = np.where(np.all(assignments[i] == unq_data, axis=1))[0]
        if len(index):
            probs[i] = counts[index[0]]
    probs = (probs + alpha) / (probs + alpha).sum()

    indicators = {var: [Bernoulli(scope=[var], p=0), Bernoulli(scope=[var], p=1)] for var in scope}

    prods = []
    for i in range(assignments.shape[0]):
        children = []
        for j in range(assignments.shape[1]):
            children.append(indicators[scope[j]][assignments[i, j]])
            # children.append(Bernoulli(scope=[scope[j]], p=assignments[i, j]))
        prods.append(Product(children=children))

    if len(prods) > 1:
        disj = Sum(children=prods, weights=probs)
    else:
        disj = prods[0]

    assign_ids(disj)
    rebuild_scopes_bottom_up(disj)

    return disj
