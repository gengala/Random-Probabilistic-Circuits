from spn.structure.Base import Sum, Product, assign_ids, rebuild_scopes_bottom_up
from spn.structure.leaves.parametric.Parametric import Bernoulli

from utils import learn_cltree, create_factors_dict
from error import RootVarError

import numpy as np


def create_cl_spn(data, scope, alpha, det=True, dtree_root=None, root_var=None):
    """ Create a Chow-Liu SPN  """

    if root_var is None:
        root_var = np.random.choice(scope)
    elif root_var not in scope:
        raise RootVarError()

    if dtree_root is None:
        dtree_root, factors_dict = learn_cltree(data, scope, root_var, alpha)
    else:
        factors_dict = create_factors_dict(data, dtree_root, alpha)

    p_neg = 0 if det else (0 + alpha) / (data.shape[0] + 2 * alpha)
    cl_spn = build_cl_spn(dtree_root, factors_dict, p_neg)

    return cl_spn


def build_cl_spn(dtree_root, factors_dict, p_neg):
    """ Build a Chow-Liu SPN given a dependency tree and its factors """

    p_pos = 1 - p_neg
    neg_buffer, pos_buffer = [], []
    nodes_stack = [dtree_root]
    last_node_visited = None

    while nodes_stack:
        """Post-Order exploration of the dependency tree"""

        node = nodes_stack[-1]

        if node.is_leaf() or (last_node_visited in node.children):

            leaves = [Bernoulli(p=p_neg, scope=[node.var_id]),
                      Bernoulli(p=p_pos, scope=[node.var_id])]

            if not node.is_leaf():

                neg_prod = Product(children=[leaves[0]] + neg_buffer[-len(node.children):])
                pos_prod = Product(children=[leaves[1]] + pos_buffer[-len(node.children):])

                del neg_buffer[-len(node.children):]
                del pos_buffer[-len(node.children):]
                sum_children = [neg_prod, pos_prod]

            else:
                sum_children = leaves

            neg_buffer.append(
                Sum(children=sum_children, weights=factors_dict[node.var_id][0]))
            pos_buffer.append(
                Sum(children=sum_children, weights=factors_dict[node.var_id][1]))

            last_node_visited = nodes_stack.pop()
        else:
            nodes_stack.extend(node.children)

    cl_spn = pos_buffer[0]
    assign_ids(cl_spn)
    rebuild_scopes_bottom_up(cl_spn)

    return cl_spn
