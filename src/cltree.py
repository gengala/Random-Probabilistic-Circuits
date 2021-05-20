from spn.structure.leaves.cltree.CLTree import CLTree

from utils import compute_probs, compute_mi_by_probs, compute_factors, build_dependency_tree_from_mi

import numpy as np


def create_cltree(data, scope, alpha, det=True, dtree_root=None, root_var=None):

    if dtree_root is None:
        node = CLTree(scope, data)
    else:
        if scope == dtree_root.get_tree()[0]:
            node = CLTree(scope, data)
        else:
            raise()

    node.det = det
    node.p_neg = (0 + alpha) / (data.shape[0] + 2 * alpha)
    update_cltree_parameters_mle(node, data, alpha, dtree_root, root_var)

    return node


def update_cltree_parameters_mle(node, data, alpha, dtree_root, root_var):

    probs, j_probs = compute_probs(data, alpha)

    if dtree_root is None:

        log_j_probs = np.log(j_probs)
        mut_info = compute_mi_by_probs(probs, j_probs, log_j_probs)
        dtree_root = build_dependency_tree_from_mi(mut_info, node.scope, root_var)

    node.tree = dtree_root.get_tree()[1]
    node.dtree_root = dtree_root

    factors = compute_factors(probs, j_probs, node.tree)

    if node.det:
        node.log_factors = np.log(factors.transpose((0, 2, 1)))
    else:
        node.weights = factors.transpose((0, 2, 1))
