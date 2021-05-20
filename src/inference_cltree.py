from spn.algorithms.Inference import add_node_likelihood
from spn.structure.leaves.cltree.CLTree import CLTree
from scipy.special import logsumexp
import numpy as np
import logging

logger = logging.getLogger(__name__)


def cltree_log_likelihood(node, data=None, dtype=np.float64, **kwargs):

    log_probs = np.zeros(data.shape[0], dtype=dtype)

    if np.isnan(np.sum(data)):
        log_factors = np.array(node.log_factors)
        for r in range(data.shape[0]):

            messages = np.zeros((node.n_features, 2))
            logprob = 0.0
            for i in node.post_order:
                state_evidence = data[r, node.scope[i]]
                if i != 0:
                    if not np.isnan(state_evidence):
                        messages[node.tree[i], 0] += (
                            log_factors[i, int(state_evidence), 0] + messages[i, int(state_evidence)]
                        )
                        messages[node.tree[i], 1] += (
                            log_factors[i, int(state_evidence), 1] + messages[i, int(state_evidence)]
                        )
                    else:
                        # marginalization
                        messages[node.tree[i], 0] += np.log(
                            np.exp(log_factors[i, 0, 0] + messages[i, 0])
                            + np.exp(log_factors[i, 1, 0] + messages[i, 1])
                        )
                        messages[node.tree[i], 1] += np.log(
                            np.exp(log_factors[i, 0, 1] + messages[i, 0])
                            + np.exp(log_factors[i, 1, 1] + messages[i, 1])
                        )
                else:
                    if not np.isnan(state_evidence):
                        logprob = log_factors[i, int(state_evidence), 0] + messages[0, int(state_evidence)]
                    else:
                        # marginalization
                        logprob = np.log(
                            np.exp(log_factors[i, 0, 0] + messages[0, 0])
                            + np.exp(log_factors[i, 1, 0] + messages[0, 1])
                        )
            log_probs[r] = logprob

    else:

        if node.det:

            vars = np.arange(node.n_features)
            scope = np.array(node.scope)
            log_probs = np.sum(node.log_factors[vars, data[:, scope], data[:, scope[node.tree]]], axis=1)

        else:

            neg_buffer, pos_buffer = [], []

            mask0 = (data[:, node.scope] == 0).T
            mask1 = ~mask0

            neg_smooth = np.empty(mask0.shape)
            pos_smooth = np.empty(mask0.shape)

            neg_smooth[mask0] = pos_smooth[mask1] = np.log(1 - node.p_neg)
            neg_smooth[mask1] = pos_smooth[mask0] = np.log(node.p_neg)

            nodes_stack = [node.dtree_root]
            last_node_visited = None
            while nodes_stack:

                peek_node = nodes_stack[-1]

                if peek_node.is_leaf() or (last_node_visited in peek_node.children):
                    feature_id = node.scope.index(peek_node.var_id)
                    n_child = len(peek_node.children)
                    if not peek_node.is_leaf():

                        neg_prod = np.sum([neg_smooth[feature_id]] + neg_buffer[-n_child:], axis=0)
                        pos_prod = np.sum([pos_smooth[feature_id]] + pos_buffer[-n_child:], axis=0)

                        del neg_buffer[-n_child:], pos_buffer[-n_child:]
                        sum_children = np.array([neg_prod, pos_prod]).T
                    else:
                        sum_children = np.array([neg_smooth[feature_id], pos_smooth[feature_id]]).T

                    neg_buffer.append(logsumexp(sum_children, b=node.weights[feature_id, :, 0], axis=1))
                    pos_buffer.append(logsumexp(sum_children, b=node.weights[feature_id, :, 1], axis=1))

                    last_node_visited = nodes_stack.pop()
                else:
                    nodes_stack.extend(peek_node.children)

            log_probs = pos_buffer[0]

    return log_probs.reshape(data.shape[0], 1)


def add_cltree_inference_support():
    add_node_likelihood(CLTree, log_lambda_func=cltree_log_likelihood)
