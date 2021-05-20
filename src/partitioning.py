import numpy as np
import itertools


class Partition:

    def __init__(self, row_ids, col_ids, super_partition=None,
                 uncond_vars=None, is_naive=False, is_conj=False):

        self.row_ids = row_ids
        self.col_ids = col_ids
        self.set_super_partition(super_partition)
        self.sub_partitions = []
        self.set_uncond_vars(uncond_vars)
        self.is_vertically_partitioned = False
        self.is_horizontally_partitioned = False
        self.is_naive = is_naive
        self.is_conj = is_conj
        self.disc_assignments = None

    def set_row_ids(self, row_ids):
        self.row_ids = row_ids

    def get_row_ids(self):
        return self.row_ids

    def set_col_ids(self, col_ids):
        self.col_ids = col_ids

    def get_col_ids(self):
        return self.col_ids

    def set_uncond_vars(self, uncond_vars):
        if uncond_vars is None:
            self.uncond_vars = self.col_ids.tolist()
            np.random.shuffle(self.uncond_vars)
        else:
            self.uncond_vars = uncond_vars

    def get_uncond_vars(self):
        return self.uncond_vars

    def set_super_partition(self, super_partition):
        self.super_partition = super_partition
        if super_partition is not None:
            super_partition.__add_sub_partition(self)

    def get_super_partition(self):
        return self.super_partition

    def __add_sub_partition(self, sub_partition):
        return self.sub_partitions.append(sub_partition)

    def get_sub_partitions(self):
        return self.sub_partitions

    def get_n_sub_partitions(self):
        return len(self.sub_partitions)

    def set_is_vertically_partitioned(self, is_vertically_partitioned):
        self.is_vertically_partitioned = is_vertically_partitioned

    def is_vertically_partitioned(self):
        return self.is_vertically_partitioned

    def set_is_horizontally_partitioned(self, is_horizontally_partitioned):
        self.is_horizontally_partitioned = is_horizontally_partitioned

    def is_horizontally_partitioned(self):
        return self.is_horizontally_partitioned

    def is_partitioned(self):
        return self.is_vertically_partitioned or self.is_horizontally_partitioned

    def get_slice(self, data):
        return data[self.row_ids][:, self.col_ids]

    def get_cond(self):
        return [col_id for col_id in self.col_ids if col_id not in self.uncond_vars]

    def set_is_naive(self, is_naive):
        self.is_naive = is_naive

    def is_naive(self):
        """ True if self have to be modeled as a naive factorization """
        return self.is_naive

    def get_cond_split(self):

        cond_split = []
        cond_vars = self.get_cond()

        if len(cond_vars) != 0 and len(cond_vars) != len(self.col_ids):
            col_ids_1 = cond_vars
            col_ids_2 = self.uncond_vars

            cond_split = [np.asarray(col_ids_1), np.asarray(col_ids_2)]

        return cond_split

    def get_rand_conj_splits(self, data, min_part_inst, conj_len, arity, str_dec):

        if len(self.get_uncond_vars()) < conj_len or len(self.row_ids) < 2 * min_part_inst:
            return [], [], []

        uncond_vars = self.get_uncond_vars().copy()

        if not str_dec:
            np.random.shuffle(uncond_vars)

        conj_vars = uncond_vars[:conj_len]
        #
        # list of all possible assignments for a conjunction with length conj_len
        assignments = [list(assignment) for assignment in itertools.product([0, 1], repeat=len(conj_vars))]
        np.random.shuffle(assignments)

        discarded_row_ids = self.row_ids.copy()
        conj_row_ids_l = []

        for assignment in assignments:

            conj = [conj_vars, assignment]
            conj_row_ids = self.get_conj_row_ids(data, conj, min_part_inst)

            if len(conj_row_ids) == len(self.row_ids):
                return [], discarded_row_ids, conj_vars.tolist()

            if len(conj_row_ids) != 0 and len(discarded_row_ids) - len(conj_row_ids) >= min_part_inst:

                discarded_row_ids = np.setdiff1d(discarded_row_ids, conj_row_ids)
                conj_row_ids_l.append(conj_row_ids)

                if len(conj_row_ids_l) == arity - 1 or len(discarded_row_ids) < 2 * min_part_inst:
                    break

        if conj_row_ids_l:
            return conj_row_ids_l, discarded_row_ids, conj_vars.tolist()
        else:
            return [], [], []

    def get_conj_row_ids(self, data, conj, min_part_inst):
        """
        It returns the row ids of the instances which satisfy a
        given conjunction iff the number of such instances is greater
        or equal than the minimum number of instances per partition;
        otherwise, it returns an empty array.

        @param conj:    is a conjunction modeled as a list of two lists.
                        The first list contains the IDs of the variables.
                        The second list contains the related assignment.
                        e.g. [[8,3],[1,0]] models the conjunction V8=1 and V3=0.
        """

        if len(self.row_ids) < min_part_inst:

            return np.empty(0, dtype=np.int32)

        else:

            conj_row_ids = self.row_ids.copy()

            for i in range(len(conj[0])):

                conj_row_ids = conj_row_ids[np.where(data[conj_row_ids, conj[0][i]] == conj[1][i])[0]]

                if len(conj_row_ids) < min_part_inst:
                    conj_row_ids = np.empty(0, dtype=np.int32)
                    break

            return conj_row_ids


def create_random_partitioning(data, str_dec, min_part_inst, conj_len, arity,  max_parts, uncond_vars):

    """ Create a random partitioning tree
    :param max_parts: maximum number of partitions to reach
    :param data: data matrix to partition
    :param min_part_inst: minimum number of instances per partition
    :param conj_len: length of the conjunctions to be used to horizontally split
    :param arity: Maximum number of subpartitions for an horizontal partitioned partition
    :param str_dec: True to ensure structured decomposability; False otherwise
    :param uncond_vars: list of variables from which to extract the first *conj_len* ones to create the conjunctions
    :return partition_root: The partition root of the tree
    :return n_partitions: number of partitions reached
    """

    partition_root = Partition(row_ids=np.arange(data.shape[0]),
                               col_ids=np.array(uncond_vars),
                               uncond_vars=np.array(uncond_vars),
                               super_partition=None)

    n_partitions = 0
    conj_vars_l = []
    cl_parts_l = []
    partition_leaves = [partition_root]

    while partition_leaves and n_partitions + len(partition_leaves) < max_parts:

        part = partition_leaves.pop(np.random.randint(len(partition_leaves)))

        conj_row_ids_l, discarded_row_ids, conj_vars = \
            part.get_rand_conj_splits(data, min_part_inst, conj_len, arity, str_dec)

        if len(discarded_row_ids):

            if conj_vars not in conj_vars_l: conj_vars_l.append(conj_vars)

            part.set_is_horizontally_partitioned(True)
            #
            # the following instruction ensures a general definition of the list
            # uncond_vars, preserving its order
            uncond_vars = [uv for uv in part.get_uncond_vars() if uv not in conj_vars]

            part_buffer = [
                Partition(row_ids=discarded_row_ids,
                          col_ids=part.get_col_ids().copy(),
                          uncond_vars=uncond_vars.copy(),
                          super_partition=part)]

            for conj_row_ids in conj_row_ids_l:
                part_buffer.append(
                    Partition(row_ids=conj_row_ids,
                              col_ids=part.get_col_ids().copy(),
                              uncond_vars=uncond_vars.copy(),
                              super_partition=part))

            discarded_assignments = \
                set([tuple(assignment) for assignment in itertools.product([0, 1], repeat=len(conj_vars))])

            for k in range(len(part_buffer)):

                part = part_buffer[k]
                cond_split = part.get_cond_split()

                if cond_split:

                    n_partitions += 1
                    part.set_is_vertically_partitioned(True)

                    is_conj = False if not k else True
                    p = Partition(row_ids=part.get_row_ids().copy(),
                                  col_ids=cond_split[0].copy(),
                                  uncond_vars=[],
                                  super_partition=part,
                                  is_naive=True,
                                  is_conj=is_conj)

                    if is_conj:
                        discarded_assignments.remove(tuple(p.get_slice(data)[0]))

                    partition_leaves.append(
                        Partition(row_ids=part.get_row_ids().copy(),
                                  col_ids=cond_split[1].copy(),
                                  uncond_vars=cond_split[1].copy(),
                                  super_partition=part))

                else:
                    partition_leaves.append(part)

            if part_buffer[0].sub_partitions:
                part_buffer[0].sub_partitions[0].disc_assignments = \
                    np.array(list(discarded_assignments))

        else:
            n_partitions += 1
            cl_parts_l.append(part)

    #
    # in case the process ended because n_partitions + len(partition_leaves) > max_parts
    n_partitions += len(partition_leaves)
    cl_parts_l.extend(partition_leaves)

    print('Partitions reached:', n_partitions)

    return partition_root, cl_parts_l, conj_vars_l, n_partitions
