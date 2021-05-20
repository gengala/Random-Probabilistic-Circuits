from spn.structure.Base import Product, Sum, get_nodes_by_type
from spn.structure.leaves.cltree.CLTree import CLTree
from spn.algorithms.Validity import is_consistent

from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import depth_first_order

from error import RootVarError

import numpy as np

ROOT = -1


class VTreeNode:
    """ used to model a Vtree """

    def __init__(self, var_id=None, children=[]):
        self.var_id = var_id
        self.parent = None
        self.innerNode = True if self.var_id is None else False
        self.set_children(children)
        self.scopes = None

    def get_var_id(self):
        return self.var_id

    def get_parent(self):
        return self.parent

    def get_children(self):
        return self.children

    def set_children(self, children):
        self.children = children
        for child in children:
            child.parent = self

    def set_scopes(self, scopes):
        self.scopes = scopes

    def get_scopes(self):
        return self.scopes

    def is_leaf(self):
        return len(self.children) == 0

    def is_inner(self):
        return len(self.children) != 0


class DTreeNode:
    """ used to model a dependency tree """

    def __init__(self, var_id, parent=None):
        self.var_id = var_id
        self.set_parent(parent)
        self.children = []
        self.tree = None

    def get_var_id(self):
        return self.var_id

    def get_parent(self):
        return self.parent

    def get_children(self):
        return self.children

    def set_parent(self, parent):
        if parent is not None:
            self.parent = parent
            self.parent.children.append(self)

    def set_tree(self, scope, tree):
        self.tree = [scope, tree]

    def get_tree(self):
        return self.tree

    def is_leaf(self):
        return len(self.children) == 0


def compute_probs(data, alpha):

    #
    # for fast np dot
    if data.dtype != np.float32:
        data = data.astype(np.float32)

    n_features = data.shape[1]
    n_samples = data.shape[0]

    j_ones = np.dot(data.T, data)
    #
    # to avoid normalization errors (weights of sum nodes have to sum up to 1)
    j_ones = j_ones.astype(np.float64)
    j_ones_diag = np.diag(j_ones)

    cols_diag = j_ones_diag * np.ones((n_features, n_features))
    rows_diag = cols_diag.transpose()

    probs = np.zeros((n_features, 2))
    j_probs = np.zeros((n_features, n_features, 2, 2))

    probs[:, 1] = (j_ones_diag + 2 * alpha) / (n_samples + 4 * alpha)
    probs[:, 0] = 1 - probs[:, 1]

    j_probs[:, :, 0, 0] = n_samples - cols_diag - rows_diag + j_ones + alpha
    j_probs[:, :, 0, 1] = cols_diag - j_ones + alpha
    j_probs[:, :, 1, 0] = rows_diag - j_ones + alpha
    j_probs[:, :, 1, 1] = j_ones + alpha

    j_probs = j_probs / (n_samples + 4 * alpha)

    return probs, j_probs


def compute_factors(probs, j_probs, tree):

    factors = np.zeros((probs.shape[0], 2, 2))

    root_id = tree.index(ROOT)

    features = np.arange(probs.shape[0]).tolist()
    features.remove(root_id)

    parents = tree.copy()
    parents.pop(root_id)

    factors[root_id, 0, 0] = factors[root_id, 1, 0] = probs[root_id, 0]
    factors[root_id, 0, 1] = factors[root_id, 1, 1] = probs[root_id, 1]

    factors[features, 0, 0] = j_probs[features, parents, 0, 0] / probs[parents, 0]
    factors[features, 1, 0] = j_probs[features, parents, 0, 1] / probs[parents, 1]
    factors[features, 0, 1] = j_probs[features, parents, 1, 0] / probs[parents, 0]
    factors[features, 1, 1] = j_probs[features, parents, 1, 1] / probs[parents, 1]

    return factors


def compute_mi_by_probs(probs, j_probs, log_j_probs):

    prod_probs = np.zeros((probs.shape[0], probs.shape[0], 2, 2))

    prod_probs[:, :, 0, 0] = np.outer(probs[:, 0], probs[:, 0])
    prod_probs[:, :, 0, 1] = np.outer(probs[:, 0], probs[:, 1])
    prod_probs[:, :, 1, 0] = np.outer(probs[:, 1], probs[:, 0])
    prod_probs[:, :, 1, 1] = np.outer(probs[:, 1], probs[:, 1])

    mut_info = np.sum(j_probs * (log_j_probs - np.log(prod_probs)), axis=(2, 3))
    np.fill_diagonal(mut_info, 0)

    return mut_info


def compute_mi(data, alpha):

    probs, j_probs = compute_probs(data, alpha)

    return compute_mi_by_probs(probs, j_probs, np.log(j_probs))


def build_dependency_tree_from_mi(mut_info, scope, root_var=None):

    if root_var is None:
        root_var = np.random.choice(scope)
    else:
        if root_var not in scope:
            raise RootVarError()

    root_id = scope.index(root_var)

    mst = minimum_spanning_tree(-(mut_info + 1))
    dfs_tree = depth_first_order(mst, directed=False, i_start=root_id)
    tree = dfs_tree[1].tolist()
    tree[root_id] = ROOT

    dtree_nodes = {var_id: DTreeNode(var_id) for var_id in scope}

    parents = np.arange(mut_info.shape[0]).tolist()
    parents.remove(root_id)

    for p in parents:
        dtree_nodes[scope[p]].set_parent(dtree_nodes[scope[tree[p]]])

    dtree_nodes[scope[root_id]].set_tree(scope, tree)

    return dtree_nodes[scope[root_id]]


def learn_cltree(data, scope, root_var, alpha):

    if root_var not in scope:
        raise RootVarError()

    probs, j_probs = compute_probs(data, alpha)

    mut_info = compute_mi_by_probs(probs, j_probs, np.log(j_probs))

    dtree_root = build_dependency_tree_from_mi(mut_info, scope, root_var)

    factors = compute_factors(probs, j_probs, dtree_root.get_tree()[1])

    factors_dict = {var: factors[scope.index(var)] for var in scope}

    return dtree_root, factors_dict


def create_factors_dict(data, dtree_root, alpha):

    probs, j_probs = compute_probs(data, alpha)

    scope = dtree_root.get_tree()[0]
    tree = dtree_root.get_tree()[1]

    factors = compute_factors(probs, j_probs, tree)
    factors_dict = {var: factors[scope.index(var)] for var in scope}

    return factors_dict


def create_dtree_dict(data_l, cl_parts_l, conj_vars_l, alpha):

    n_vars = data_l[0].shape[1]

    sum_mut_info = np.zeros((n_vars, n_vars))
    for i in range(len(data_l)):
        for part in cl_parts_l[i]:
            mi = compute_mi(part.get_slice(data_l[i]), alpha)
            sum_mut_info[part.col_ids[:, None], part.col_ids] += mi

    #
    # create a dependency tree for each scope in scopes
    free_vars = list(set(np.arange(n_vars))-set([var for conj_vars in conj_vars_l for var in conj_vars]))

    if free_vars:
        scopes = conj_vars_l + [free_vars]
    else:
        scopes = conj_vars_l

    dtrees = []
    for scope in scopes:
        dtrees.append(build_dependency_tree_from_mi(sum_mut_info[scope][:, scope], scope))

    scope = dtrees[-1].get_tree()[0].copy()
    tree = dtrees[-1].get_tree()[1].copy()

    #
    # concatenate dtrees
    for k in reversed(range(0, len(dtrees) - 1)):

        tree += [t + len(scope) if t != ROOT else t for t in dtrees[k].get_tree()[1]]
        tree[tree.index(ROOT)] = tree.index(ROOT, len(scope))
        scope += dtrees[k].get_tree()[0]

        dtrees[k].set_tree(scope.copy(), tree.copy())
        dtrees[k + 1].set_parent(dtrees[k])

    #
    # return a dictionary of dtrees where keys are scope lengths
    return {len(dtree.get_tree()[0]): dtree for dtree in dtrees}


def is_structured_decomposable(spn, verbose=False):

    if not is_consistent(spn):
        return False

    nodes = get_nodes_by_type(spn)

    scope_set = set()
    for n in nodes:
        if isinstance(n, Product):
            scope_set.add(tuple(n.scope))
        elif isinstance(n, CLTree):
            vtree = from_dtree_to_vtree(n.dtree_root)
            scope_set.update([tuple(s) for s in vtree.scopes])

    scopes = list(scope_set)
    scopes = [set(t) for t in scopes]
    #
    # ordering is not needed, but useful for printing
    if verbose:
        scopes.sort(key=len)
        for s in scopes:
            print(s)

    for i in range(len(scopes)):
        for j in range(len(scopes)):
            int_len = len(scopes[i].intersection(scopes[j]))
            if int_len != 0 and int_len != min(len(scopes[i]), len(scopes[j])):
                return False

    return True


def circuit_size(spn):

    nodes = get_nodes_by_type(spn)
    size = 0
    for n in nodes:
        if isinstance(n, Product) or isinstance(n, Sum):
            size += len(n.children)
        elif isinstance(n, CLTree):
            queue = [n.dtree_root]
            clt_size = 0
            while queue:
                peek_node = queue.pop(0)
                queue.extend(peek_node.children)
                if not peek_node.is_leaf():
                    clt_size += (1 + len(peek_node.children)) * 2
                clt_size += 4
            size += clt_size - 2

    return size


def from_dtree_to_vtree(dtree_root):

    if len(dtree_root.get_tree()[0]) == 1:
        vtree_root = VTreeNode(var_id=dtree_root.get_tree()[0][0])
        vtree_root.set_scopes([dtree_root.get_tree()[0]])
        return vtree_root

    scopes = []
    scopes_t = []

    last_node_visited = None
    dtree_stack = [dtree_root]
    buffer = []

    while dtree_stack:

        peek_dnode = dtree_stack[-1]

        if not peek_dnode.children or (last_node_visited in peek_dnode.children):

            if not peek_dnode.is_leaf():

                leaves = []

                n_parents = 0
                for child in peek_dnode.children:
                    if child.is_leaf():
                        leaves.append(VTreeNode(child.var_id))
                    else:
                        n_parents += 1

                if n_parents:
                    temp_buffer = buffer[-n_parents:]
                    del buffer[-n_parents:]
                else:
                    temp_buffer = []

                vtree_root = VTreeNode(children=leaves + temp_buffer + [VTreeNode(peek_dnode.var_id)])

                #
                # this piece of code store all the scopes of the vtree
                # ------
                scope = []
                n_inner_children = 0
                for n in vtree_root.children:
                    if not n.is_inner():
                        scope.append(n.var_id)
                    else:
                        n_inner_children += 1
                if n_inner_children:
                    prev_scopes = scopes_t[-n_inner_children:]
                    del scopes_t[-n_inner_children:]
                    scope.extend([v for s in prev_scopes for v in s])
                scopes_t.append(scope)
                scopes.append(scope)
                # ------

                buffer.append(vtree_root)

            dtree_stack.pop()
            last_node_visited = peek_dnode

        else:

            dtree_stack.extend(peek_dnode.children)

    vtree_root.set_scopes(scopes)

    return vtree_root
