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

from scipy.special import logsumexp

from src.inference import log_likelihood

from src.xpc import create_expc
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

parser.add_argument('-dim', '--ensemble-dimension', type=int, nargs='+',
                    default=[10],
                    help='EXPC dimension')

parser.add_argument('-sd', '--str-dec-level', type=int, nargs='+',
                    default=[0],
                    help='0 for no-SD EXPC; 1 for no-SD EXPC with SD XPCs; 2 for SD EXPC')

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
ensemble_dim_l = args.ensemble_dimension
sd_level_l = args.str_dec_level
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

out_single_model_path = out_path + '/single_models.lls'
out_ensemble_path = out_path + '/ensembles.lls'

#
# creating dir if non-existing
if not os.path.exists(os.path.dirname(out_single_model_path)):
    os.makedirs(os.path.dirname(out_single_model_path))

logging.basicConfig(filename=out_path + '/exp.log', level=logging.INFO)
logging.info("Starting with arguments:\n%s", args)
# I shall print here all the stats

#
# elaborating the dataset
logging.info('Loading dataset: %s..', dataset_name)
train, valid, test = dataset.load_train_val_test_csvs(dataset_name)

preamble_1 = "runs\tsd-level\tdet-level\t" \
             "min-part-inst\tconj-len\tarity\tmax-parts\tsmoothing\t"\
             "avg-spn-train-time\t" \
             "avg-n-parts\tstd-n-parts\t" \
             "avg-circuit-sizes\tstd-circuit-sizes\t" \
             "avg-avg-valid-lls\tstd-avg-valid-lls\t" \
             "avg-avg-test-lls\tstd-avg-test-lls\t" \
             "best-spn-avg-test-ll\n"

preamble_2 = "ensemble-dim\tsd-level\tdet-level\t" \
             "min-part-inst\tconj-len\tarity\tmax-parts\tsmoothing\t"\
             "ensemble-spn-avg-valid-ll\tensemble-spn-avg-test-ll\n"

out_ensemble = open(out_ensemble_path, 'w')
with open(out_single_model_path, 'w') as out_single_model:

    heading = "parameters:\n{0}\n\n".format(args)
    heading = heading.replace('ensemble_dimension=%s' % str(ensemble_dim_l), 'runs=[%s]' % str(max(ensemble_dim_l)))
    out_single_model.write(heading)
    out_single_model.write(preamble_1)
    out_single_model.flush()

    out_ensemble.write("parameters:\n{0}\n\n".format(args))
    out_ensemble.write(preamble_2)
    out_ensemble.flush()

    total_combinations = np.prod([len(sd_level_l), len(det_level_l), len(min_part_inst_l),
                                  len(conj_len_l), len(arity_l), len(max_parts_l), len(alpha_smoothing_l)])

    n_SPNs = int(np.max(ensemble_dim_l))

    comb_counter = 1

    #
    # looping over all parameters combinations
    for sd_level in sd_level_l:
        for det_level in det_level_l:
            for min_part_inst in min_part_inst_l:
                for conj_len in conj_len_l:
                    for arity in arity_l:
                        for max_parts in max_parts_l:
                            for alpha_smoothing in alpha_smoothing_l:

                                combination_string = 'ds=%s, sd=%s, det=%s, m=%s, l=%s, a=%s, p=%s, s=%s (%s/%s)' % \
                                                     (dataset_name, sd_level, det_level, min_part_inst,
                                                      conj_len, arity, max_parts, alpha_smoothing, comb_counter, total_combinations)

                                print(combination_string)
                                logging.info('Combination: %s' % combination_string)

                                try:

                                    #
                                    # Start training
                                    train_start_t = perf_counter()
                                    ensemble_spn, n_parts = \
                                        create_expc(data=train,
                                                    ensemble_dim=n_SPNs,
                                                    sd_level=sd_level,
                                                    det_level=det_level,
                                                    min_part_inst=min_part_inst,
                                                    conj_len=conj_len,
                                                    arity=arity,
                                                    leaves=create_cltree,
                                                    alpha=alpha_smoothing,
                                                    max_parts=max_parts)
                                    train_end_t = perf_counter()
                                    train_t = train_end_t - train_start_t
                                    #
                                    # End training

                                    #
                                    # Start validating
                                    valid_lls = np.zeros((valid.shape[0], n_SPNs))
                                    valid_start_t = perf_counter()
                                    for k in range(n_SPNs):
                                        print('Validating XPC_%s/%s' % (k, n_SPNs))
                                        valid_lls[:, k] = log_likelihood(ensemble_spn.children[k], valid)[:, 0]
                                    valid_end_t = perf_counter()
                                    valid_t = valid_end_t - valid_start_t
                                    #
                                    # End validating

                                    #
                                    # Start testing
                                    test_lls = np.zeros((test.shape[0], n_SPNs))
                                    test_start_t = perf_counter()
                                    for k in range(n_SPNs):
                                        print('Testing XPC_%s/%s' % (k, n_SPNs))
                                        test_lls[:, k] = log_likelihood(ensemble_spn.children[k], test)[:, 0]
                                    test_end_t = perf_counter()
                                    test_t = test_end_t - test_start_t
                                    #
                                    # End testing

                                    #
                                    # Start computing metrics
                                    avg_spn_train_t = train_t / n_SPNs

                                    avg_n_parts = np.mean(n_parts)
                                    std_n_parts = np.std(n_parts)

                                    circuit_sizes = np.zeros(n_SPNs)
                                    for k in range(n_SPNs):
                                        circuit_sizes[k] = circuit_size(ensemble_spn.children[k])
                                    avg_circuit_sizes = np.mean(circuit_sizes)
                                    std_circuit_sizes = np.std(circuit_sizes)

                                    avg_valid_lls = np.zeros(n_SPNs)
                                    for k in range(n_SPNs):
                                        avg_valid_lls[k] = np.mean(valid_lls[:, k])
                                    avg_avg_valid_lls = np.mean(avg_valid_lls)
                                    std_avg_valid_lls = np.std(avg_valid_lls)

                                    avg_test_lls = np.zeros(n_SPNs)
                                    for k in range(n_SPNs):
                                        avg_test_lls[k] = np.mean(test_lls[:, k])
                                    avg_avg_test_lls = np.mean(avg_test_lls)
                                    std_avg_test_lls = np.std(avg_test_lls)

                                    best_spn_avg_test_ll = avg_test_lls[np.argmax(avg_valid_lls)]
                                    #
                                    # End computing metrics

                                    #
                                    # Write to file a line for the grid
                                    stats = stats_format([n_SPNs,
                                                          sd_level,
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

                                    out_single_model.write(stats + '\n')
                                    out_single_model.flush()

                                    for ensemble_dim in ensemble_dim_l:
                                        valid_lls_ = valid_lls[:, :ensemble_dim]
                                        test_lls_ = test_lls[:, :ensemble_dim]

                                        ensemble_spn_valid_ll = logsumexp(valid_lls_,
                                                                          b=np.full(ensemble_dim, 1 / ensemble_dim), axis=1)
                                        ensemble_spn_test_ll = logsumexp(test_lls_,
                                                                         b=np.full(ensemble_dim, 1 / ensemble_dim), axis=1)

                                        ensemble_spn_avg_valid_ll = np.mean(ensemble_spn_valid_ll)
                                        ensemble_spn_avg_test_ll = np.mean(ensemble_spn_test_ll)

                                        #
                                        # Write to file a line for the grid
                                        stats = stats_format([ensemble_dim,
                                                              sd_level,
                                                              det_level,
                                                              min_part_inst,
                                                              conj_len,
                                                              arity,
                                                              max_parts,
                                                              alpha_smoothing,
                                                              ensemble_spn_avg_valid_ll,
                                                              ensemble_spn_avg_test_ll],
                                                             '\t',
                                                             digits=5)

                                        out_ensemble.write(stats + '\n')
                                        out_ensemble.flush()

                                except Error as err:
                                    logging.info(err)
                                    logging.info('Discarded combination')
                                except Exception as err:
                                    logging.exception(err)
                                finally:
                                    comb_counter += 1

    out_single_model.close()
    out_ensemble.close()

print('Grid search ended on ' + dataset_name)
logging.info('Grid search ended on ' + dataset_name)
