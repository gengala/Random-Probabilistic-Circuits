from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up, Sum, Product

from src.partitioning import create_random_partitioning
from src.naive_fact import create_naive_fact, create_conj, create_disj
from src.cltree import create_cltree
from src.cl_spn import create_cl_spn

from utils import create_dtree_dict, compute_mi
from error import StructDecError, ArityError, NoRandomness, NoPartitioningFound, DetError

import numpy as np

SD_LEVEL_0 = 0  # N-SD
SD_LEVEL_1 = 1  # X-SD
SD_LEVEL_2 = 2  # SD
SD_LEVELS = [SD_LEVEL_0, SD_LEVEL_1, SD_LEVEL_2]

DET_LEVEL_0 = 0  # N-Det
DET_LEVEL_1 = 1  # Det
DET_LEVELS = [DET_LEVEL_0, DET_LEVEL_1]


def create_leaf_pc(data, part, leaves, dtree_dict, det_level, alpha):

    if part.is_conj:

        if det_level == DET_LEVEL_0:
            leaf_pc = create_conj(data=part.get_slice(data), scope=part.col_ids.tolist(), alpha=alpha)
        else:
            leaf_pc = create_conj(data=part.get_slice(data), scope=part.col_ids.tolist(), alpha=0)

    elif part.is_naive:

        if det_level == DET_LEVEL_0:
            leaf_pc = create_naive_fact(data=part.get_slice(data), scope=part.get_col_ids().tolist(), alpha=alpha)
        elif part.disc_assignments.shape[0] == 2 ** part.disc_assignments.shape[1]:
            leaf_pc = create_naive_fact(data=part.get_slice(data), scope=part.get_col_ids().tolist(), alpha=alpha)
        else:
            leaf_pc = create_disj(data=part.get_slice(data), scope=part.col_ids.tolist(), assignments=part.disc_assignments, alpha=alpha)

    elif leaves == create_cl_spn or leaves == create_cltree:

        if dtree_dict is not None:
            dtree_root = dtree_dict[len(part.col_ids)]
            scope = dtree_root.get_tree()[0]
            data_slice = data[part.row_ids][:, scope]
        else:
            dtree_root = None
            scope = part.col_ids.tolist()
            data_slice = part.get_slice(data)

        det = (det_level == DET_LEVEL_1)
        leaf_pc = leaves(data=data_slice, scope=scope, det=det, dtree_root=dtree_root, alpha=alpha)

    elif leaves == create_naive_fact:

        leaf_pc = create_naive_fact(data=part.get_slice(data), scope=part.col_ids.tolist(), alpha=alpha)

    return leaf_pc


def build_xpc_bottom_up(data, part_root, dtree_dict, det_level, leaves, alpha):
    """
    Build the XPC induced by the partitions tree in a bottom up way.
     The building process is based on the post-order traversal exploration of the partitions tree
    :param alpha: smoothing factor
    :param det_level: 0 for non det., 1 for relaxed det. and 2 for det.
    :param dtree_dict: None if no dependency tree has to be respected, a dictionary of dtree otherwise
    :param part_root: A random partitions tree
    :param data: The data to model
    :param leaves: multivariate leaf function
    :return: the XPC induced by the partition tree
    """

    partitions_stack = [part_root]
    pc_nodes_stack = []
    last_part_exp = None

    while partitions_stack:

        part = partitions_stack[-1]

        if not part.is_partitioned() or (last_part_exp in part.get_sub_partitions()):

            if part.is_partitioned():

                pc_child_nodes = pc_nodes_stack[-len(part.get_sub_partitions()):]
                pc_nodes_stack = pc_nodes_stack[:len(pc_nodes_stack) - len(part.get_sub_partitions())]

                if part.is_horizontally_partitioned:

                    weights = [len(sub_part.row_ids)/len(part.row_ids) for sub_part in part.get_sub_partitions()]
                    pc_sum_node = Sum(weights=weights, children=pc_child_nodes)
                    pc_nodes_stack.append(pc_sum_node)

                else:

                    pc_child_nodes_ = []
                    for c in pc_child_nodes:
                        if isinstance(c, Product) or (isinstance(c, Sum) and len(c.children) == 1):
                            pc_child_nodes_.extend(c.children)
                        else:
                            pc_child_nodes_.append(c)
                    pc_prod_node = Product(children=pc_child_nodes_)
                    pc_nodes_stack.append(pc_prod_node)

            else:

                leaf_pc = create_leaf_pc(data, part, leaves, dtree_dict, det_level, alpha)
                pc_nodes_stack.append(leaf_pc)

            last_part_exp = partitions_stack.pop()
        else:
            partitions_stack.extend(part.get_sub_partitions()[::-1])

    xpc = pc_nodes_stack[0]
    assign_ids(xpc)
    rebuild_scopes_bottom_up(xpc)

    return xpc


def greedy_vars_ordering(data, conj_len):

    mut_info = compute_mi(data, 0.1)
    sums = np.sum(mut_info, axis=0)
    uncond_vars = []

    free_vars = np.arange(data.shape[1]).tolist()

    while free_vars:

        peek_var = free_vars[np.argmax(sums[free_vars])]

        free_vars.remove(peek_var)
        uncond_vars.append(peek_var)

        if len(free_vars) > conj_len - 1:
            idx = np.argpartition(-mut_info[peek_var][free_vars], conj_len - 1)[:conj_len - 1]
            vars = np.array(free_vars)[idx].tolist()
        else:
            vars = free_vars.copy()

        free_vars = list(set(free_vars) - set(vars))
        uncond_vars.extend(vars)

    return uncond_vars


def create_expc(data, ensemble_dim, sd_level, det_level, min_part_inst, conj_len, arity, leaves,
                alpha=0.01, bagging=False, max_parts=1000, random_seed=42):

    if sd_level not in SD_LEVELS:
        raise StructDecError()

    if det_level not in DET_LEVELS:
        raise DetError()

    if arity < 2 or arity > 2 ** conj_len:
        raise ArityError()

    if sd_level == SD_LEVEL_2 and conj_len == 1:
        raise NoRandomness()

    print('Generating random partitionings..')
    np.random.seed(random_seed)

    str_dec = (sd_level == SD_LEVEL_1 or sd_level == SD_LEVEL_2)

    if sd_level == SD_LEVEL_2:
        uncond_vars = greedy_vars_ordering(data, conj_len)
    else:
        uncond_vars = np.arange(data.shape[1]).tolist()

    conj_vars_l = [None] * ensemble_dim
    cl_parts_l = [None] * ensemble_dim
    n_parts_l = [None] * ensemble_dim
    ptrees_l = [None] * ensemble_dim
    data_l = [None] * ensemble_dim
    xpc_l = [None] * ensemble_dim

    for i in range(ensemble_dim):

        if sd_level != SD_LEVEL_2:
            np.random.shuffle(uncond_vars)

        if bagging:
            data_l[i] = data[np.random.choice(a=data.shape[0], size=data.shape[0] * 70 // 100, replace=True)]
        else:
            data_l[i] = data

        print(uncond_vars)
        ptrees_l[i], cl_parts_l[i], conj_vars_l[i], n_parts_l[i] = \
            create_random_partitioning(data=data_l[i],
                                       str_dec=str_dec,
                                       min_part_inst=min_part_inst,
                                       conj_len=conj_len,
                                       arity=arity,
                                       max_parts=max_parts,
                                       uncond_vars=uncond_vars)

    if all([n_parts == 1 for n_parts in n_parts_l]):
        raise NoPartitioningFound()

    if sd_level == SD_LEVEL_0 or leaves == create_naive_fact:

        dtree_dict = None
        for i in range(ensemble_dim):
            print('Learning XPC_%s/%s' % (i, ensemble_dim))
            xpc_l[i] = build_xpc_bottom_up(data_l[i], ptrees_l[i], dtree_dict, det_level, leaves, alpha)

    elif sd_level == SD_LEVEL_1:

        for i in range(ensemble_dim):
            print('Learning XPC_%s/%s' % (i, ensemble_dim))
            #
            # learn a dtree for each XPC
            dtree_dict = create_dtree_dict([data_l[i]], [cl_parts_l[i]], conj_vars_l[i], alpha)
            xpc_l[i] = build_xpc_bottom_up(data_l[i], ptrees_l[i], dtree_dict, det_level, leaves, alpha)

    elif sd_level == SD_LEVEL_2:

        #
        # learn a dtree for the ensemble
        print('Learning a dependency tree for the ensemble..')
        dtree_dict = create_dtree_dict(data_l, cl_parts_l, max(conj_vars_l, key=len), alpha)
        for i in range(ensemble_dim):
            print('Building XPC_%s/%s' % (i, ensemble_dim))
            xpc_l[i] = build_xpc_bottom_up(data_l[i], ptrees_l[i], dtree_dict, det_level, leaves, alpha)

    expc = Sum(weights=1 / ensemble_dim * np.ones(ensemble_dim), children=xpc_l)
    assign_ids(expc)
    rebuild_scopes_bottom_up(expc)

    return expc, n_parts_l


def create_xpc(data, sd_level, det_level, min_part_inst, conj_len, arity,
               leaves, alpha=0.01, max_parts=1000, random_seed=42):

    if sd_level not in SD_LEVELS:
        raise StructDecError()

    if det_level not in DET_LEVELS:
        raise DetError()

    if arity < 2 or arity > 2 ** conj_len:
        raise ArityError()

    np.random.seed(random_seed)

    if sd_level == SD_LEVEL_2:
        uncond_vars = greedy_vars_ordering(data, conj_len)
    else:
        uncond_vars = np.arange(data.shape[1]).tolist()
        np.random.shuffle(uncond_vars)

    str_dec = (sd_level == SD_LEVEL_1 or sd_level == SD_LEVEL_2)

    part_root, cl_parts_l, conj_vars_l, n_parts = \
        create_random_partitioning(data=data,
                                   str_dec=str_dec,
                                   min_part_inst=min_part_inst,
                                   conj_len=conj_len,
                                   arity=arity,
                                   max_parts=max_parts,
                                   uncond_vars=uncond_vars)

    if n_parts == 1:
        raise NoPartitioningFound()

    if (sd_level == SD_LEVEL_1 or sd_level == SD_LEVEL_2) and leaves != create_naive_fact:
        dtree_dict = create_dtree_dict([data], [cl_parts_l], conj_vars_l, alpha)
    else:
        dtree_dict = None

    xpc = build_xpc_bottom_up(data, part_root, dtree_dict, det_level, leaves, alpha)

    return xpc, n_parts
