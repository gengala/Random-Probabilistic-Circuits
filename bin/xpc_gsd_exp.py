import argparse

try:
    from time import perf_counter
except:
    from time import time

    perf_counter = time

import dataset

import numpy as np

import datetime

import os

import logging

from src.inference import log_likelihood

from src.xpc import create_xpc, SD_LEVEL_2
from src.cltree import create_cltree
from utils import circuit_size

from error import Error, NoPartitioningFound


def stats_format(stats_list, separator, digits=5):
    formatted = []
    float_format = '{0:.' + str(digits) + 'f}'
    for stat in stats_list:
        if isinstance(stat, int):
            formatted.append(str(stat))
        elif isinstance(stat, float):

            formatted.append(float_format.format(stat))
        else:
            formatted.append(stat)
    # concatenation
    return separator.join(formatted)


#########################################
# creating the opt parser

parser = argparse.ArgumentParser()

parser.add_argument("dataset", type=str, nargs=1,
                    help='Specify a dataset name from data (e.g. nltcs)')

parser.add_argument('-r', '--runs', type=int, nargs=1,
                    default=10,
                    help='Number of runs for each configuration')

parser.add_argument('-det', '--determinism', type=int, nargs='+',
                    default=[0],
                    help='0 for no determinism; 1 for determinism')

parser.add_argument('-m', '--min-partition-instances', type=int, nargs='+',
                    default=[256],
                    help='Minimum number of instances per partition')

parser.add_argument('-l', '--conjunction-length', type=int, nargs='+',
                    default=[2],
                    help='Conjunction length')

parser.add_argument('-a', '--arity', type=int, nargs='+',
                    default=[2],
                    help='Maximum number of sum nodes children')

parser.add_argument('-p', '--max-partitions', type=int, nargs='+',
                    default=[1000],
                    help='Maximum number of leaf partitions')

parser.add_argument('-s', '--smoothing', type=float, nargs='+',
                    default=[0.01],
                    help='Smoothing parameter alpha')

parser.add_argument('-o', '--output', type=str, nargs='?',
                    default='./exp/',
                    help='Output dir path')

#
# parsing the args
args = parser.parse_args()

#
# gathering args
runs = args.runs[0]
det_level_l = args.determinism
min_part_inst_l = args.min_partition_instances
conj_len_l = args.conjunction_length
arity_l = args.arity
max_parts_l = args.max_partitions
alpha_smoothing_l = args.smoothing

output = args.output
(dataset_name,) = args.dataset

#
# Opening the file for test prediction
date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_path = output + dataset_name + '_' + date_string

out_xpc_gsd_path = out_path + '/xpc_gsd.lls'

#
# creating dir if non-existing
if not os.path.exists(os.path.dirname(out_xpc_gsd_path)):
    os.makedirs(os.path.dirname(out_xpc_gsd_path))

logging.basicConfig(filename=out_path + '/exp.log', level=logging.INFO)
logging.info("Starting with arguments:\n%s", args)
# I shall print here all the stats

#
# elaborating the dataset
logging.info('Loading dataset: %s..', dataset_name)
train, valid, test = dataset.load_train_val_test_csvs(dataset_name)

preamble = "runs\tdet-level\tmin-part-inst\t" \
           "conj-len\tarity\tmax-parts\tsmoothing\t" \
           "avg-spn-train-time\t" \
           "avg-n-parts\tstd-n-parts\t" \
           "avg-circuit-sizes\tstd-circuit-sizes\t" \
           "avg-avg-valid-lls\tstd-avg-valid-lls\t" \
           "avg-avg-test-lls\tstd-avg-test-lls\t" \
           "best-spn-avg-test-ll\n"

exp_start_t = perf_counter()
with open(out_xpc_gsd_path, 'w') as out_xpc_gsd:

    out_xpc_gsd.write("parameters:\n{0}\n\n".format(args))
    out_xpc_gsd.write(preamble)
    out_xpc_gsd.flush()

    total_combinations = np.prod([len(det_level_l), len(min_part_inst_l), len(conj_len_l),
                                  len(arity_l), len(max_parts_l), len(alpha_smoothing_l)])

    comb_counter = 1
    #
    # looping over all parameters combinations
    for det_level in det_level_l:
        for min_part_inst in min_part_inst_l:
            for conj_len in conj_len_l:
                for arity in arity_l:
                    for max_parts in max_parts_l:
                        for alpha_smoothing in alpha_smoothing_l:

                            combination_string = 'ds=%s, det=%s, m=%s, l=%s, a=%s, p=%s, s=%s, (%s/%s)' % \
                                                 (dataset_name, det_level, min_part_inst, conj_len, arity,
                                                  max_parts, alpha_smoothing, comb_counter, total_combinations)

                            print(combination_string)
                            logging.info('Combination: %s' % combination_string)

                            try:

                                #
                                # Start training
                                xpc_gsd_l = [None] * runs
                                n_parts_l = [None] * runs
                                train_start_t = perf_counter()
                                for k in range(runs):
                                    xpc_gsd_l[k], n_parts_l[k] = \
                                        create_xpc(data=train,
                                                   sd_level=SD_LEVEL_2,
                                                   det_level=det_level,
                                                   min_part_inst=min_part_inst,
                                                   conj_len=conj_len,
                                                   arity=arity,
                                                   leaves=create_cltree,
                                                   alpha=alpha_smoothing,
                                                   max_parts=max_parts,
                                                   random_seed=k)
                                train_end_t = perf_counter()
                                train_t = train_end_t - train_start_t
                                #
                                # End training

                                #
                                # Start validating
                                valid_lls = np.zeros((valid.shape[0], runs))
                                valid_start_t = perf_counter()
                                for k in range(runs):
                                    print('Validating XPC_%s/%s' % (k, runs))
                                    valid_lls[:, k] = log_likelihood(xpc_gsd_l[k], valid)[:, 0]
                                valid_end_t = perf_counter()
                                valid_t = valid_end_t - valid_start_t
                                #
                                # End validating

                                #
                                # Start testing
                                test_lls = np.zeros((test.shape[0], runs))
                                test_start_t = perf_counter()
                                for k in range(runs):
                                    print('Testing XPC_%s/%s' % (k, runs))
                                    test_lls[:, k] = log_likelihood(xpc_gsd_l[k], test)[:, 0]
                                test_end_t = perf_counter()
                                test_t = test_end_t - test_start_t
                                #
                                # End testing

                                #
                                # Start computing metrics
                                avg_spn_train_t = train_t / runs

                                avg_n_parts = np.mean(n_parts_l)
                                std_n_parts = np.std(n_parts_l)

                                circuit_sizes = np.zeros(runs)
                                for k in range(runs):
                                    circuit_sizes[k] = circuit_size(xpc_gsd_l[k])
                                avg_circuit_sizes = np.mean(circuit_sizes)
                                std_circuit_sizes = np.std(circuit_sizes)

                                avg_valid_lls = np.zeros(runs)
                                for k in range(runs):
                                    avg_valid_lls[k] = np.mean(valid_lls[:, k])
                                avg_avg_valid_lls = np.mean(avg_valid_lls)
                                std_avg_valid_lls = np.std(avg_valid_lls)

                                avg_test_lls = np.zeros(runs)
                                for k in range(runs):
                                    avg_test_lls[k] = np.mean(test_lls[:, k])
                                avg_avg_test_lls = np.mean(avg_test_lls)
                                std_avg_test_lls = np.std(avg_test_lls)

                                best_spn_avg_test_ll = avg_test_lls[np.argmax(avg_valid_lls)]
                                #
                                # End computing metrics

                                #
                                # Write to file a line for the grid
                                stats = stats_format([runs,
                                                      det_level,
                                                      min_part_inst,
                                                      conj_len,
                                                      arity,
                                                      max_parts,
                                                      alpha_smoothing,
                                                      avg_spn_train_t,
                                                      avg_n_parts,
                                                      std_n_parts,
                                                      avg_circuit_sizes,
                                                      std_circuit_sizes,
                                                      avg_avg_valid_lls,
                                                      std_avg_valid_lls,
                                                      avg_avg_test_lls,
                                                      std_avg_test_lls,
                                                      best_spn_avg_test_ll],
                                                     '\t',
                                                     digits=5)

                                out_xpc_gsd.write(stats + '\n')
                                out_xpc_gsd.flush()

                            except Error as err:
                                logging.info(err)
                                logging.info('Discarded combination')
                            except Exception as err:
                                logging.exception(err)
                            finally:
                                comb_counter += 1

    exp_end_t = perf_counter()
    out_xpc_gsd.close()

print('Grid search ended on ' + dataset_name)
logging.info('Grid search ended on ' + dataset_name)
