from pathlib import Path
import numpy
import csv

DATA_PATH = str(Path(__file__).parent) + '/data/'

DATASET_NAMES = ['nltcs',
                 'msnbc',
                 'kdd',
                 'plants',
                 'baudio',
                 'jester',
                 'bnetflix',
                 'accidents',
                 'tretail',
                 'pumsb_star',
                 'dna',
                 'kosarek',
                 'msweb',
                 'book',
                 'tmovie',
                 'cwebkb',
                 'cr52',
                 'c20ng',
                 'bbc',
                 'ad']

DATASET_NAMES_DICT = {'nltcs': 'nltcs',
                      'msnbc': 'msnbc',
                      'kdd': 'kdd',
                      'plants': 'plants',
                      'baudio': 'audio',
                      'jester': 'jester',
                      'bnetflix': 'netflix',
                      'accidents': 'accidents',
                      'tretail': 'retail',
                      'pumsb_star': 'pumsb-star',
                      'dna': 'dna',
                      'kosarek': 'kosarek',
                      'msweb': 'msweb',
                      'book': 'book',
                      'tmovie': 'eachmovie',
                      'cwebkb': 'webkb',
                      'cr52': 'routers-52',
                      'c20ng': '20news-grp',
                      'bbc': 'bbc',
                      'ad': 'ad'}


def csv_2_numpy(file, path=DATA_PATH, sep=',', type='int8'):

    file_path = path + file
    reader = csv.reader(open(file_path, "r"), delimiter=sep)
    x = list(reader)
    dataset = numpy.array(x).astype(type)
    return dataset


def load_train_val_test_csvs(dataset,
                             path=DATA_PATH,
                             sep=',',
                             type='int32',
                             suffixes=['.ts.data', '.valid.data', '.test.data']):

    csv_files = [dataset + ext for ext in suffixes]
    return [csv_2_numpy(file, path, sep, type) for file in csv_files]
