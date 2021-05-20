# Random Probabilistic Circuits

## Usage

To run the algorithms and their grid search check the scripts in the `bin/` folder.  
To learn a default ensemble from the training set portion of the `nltcs`
data you can call:

    ipython -- bin/expc_exp.py nltcs

To get an overview of the possible parameters use `ipython -- bin/expc_exp.py -h
`:

      -dim ENSEMBLE_DIMENSION [ENSEMBLE_DIMENSION ...], --ensemble-dimension ENSEMBLE_DIMENSION [ENSEMBLE_DIMENSION ...]
                            EXPC dimension
      -sd STR_DEC_LEVEL [STR_DEC_LEVEL ...], --str-dec-level STR_DEC_LEVEL [STR_DEC_LEVEL ...]
                            0 for no-SD; 1 for no-SD EXPC with SD XPCs;
                            2 for SD EXPC
      -det DETERMINISM [DETERMINISM ...], --determinism DETERMINISM [DETERMINISM ...]
                            0 for no-determinism; 1 for determinism
      -m MIN_PARTITION_INSTANCES [MIN_PARTITION_INSTANCES ...], --min-partition-instances MIN_PARTITION_INSTANCES [MIN_PARTITION_INSTANCES ...]
                            Minimum number of instances per partition
      -l CONJUNCTION_LENGTH [CONJUNCTION_LENGTH ...], --conjunction-length CONJUNCTION_LENGTH [CONJUNCTION_LENGTH ...]
                            Conjunction length
      -a ARITY [ARITY ...], --arity ARITY [ARITY ...]
                            Maximum number of sum nodes children
      -p MAX_PARTITIONS [MAX_PARTITIONS ...], --max-partitions MAX_PARTITIONS [MAX_PARTITIONS ...]
                            Maximum number of leaf partitions
      -p SMOOTHING [SMOOTHING ...], --smoothing SMOOTHING [SMOOTHING ...]
                            Smoothing parameter alpha
      -o [OUTPUT], --output [OUTPUT]
                            Output dir path



To run a grid search you can do:

    ipython -- bin/expc_exp.py nltcs -dim 10 20 -sd 0 1 2 -det 0 1 -m 256 -l 3 -a 3 -p 200

## Reference
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4775258.svg)](https://doi.org/10.5281/zenodo.4775258)
```bibtex
@InProceedings{di-mauro_649,
    title = {Random Probabilistic Circuits},
    author = {Di Mauro, Nicola and Gala, Gennaro and Iannotta, Marco and Basile, Teresa M.A.},
    booktitle = {37th Conference on Uncertainty in Artificial Intelligence (UAI) },
    year = {2021}
}
```

