# The MTIST Platform
_**m**icrobiome **ti**me **s**eries **t**est standard dataset_

MTIST is a standardized test dataset designed to benchmark microbial ecosystem inference algorithms. In this repository, we provide both the code used to generate MTIST _and_ instructions of how to benchmark an algorithm using MTIST. 

# Installation
Install with `pip` in "editable" mode.

1. Clone repo
2. Navigate to folder
3. Create virtual environment
4. Install in editable mode using `pip install -e .`

Project will soon be uploaded to PyPi and BioConda.


# Usage

## Benchmarking using MTIST


### Benchmarking an algorithm manually
To manually benchmark an inference algorithm, run inference on each MTIST dataset and calculate ES score for each inferred community matrix.

The MTIST datasets can be found in `mtist1.0/mtist_datasets`. The metadata detailing which ground truth community matrix was used to generate each mtist dataset can be found at `mtist1.0/mtist_datasets/mtist_metadata.csv`. 

For each MTIST dataset, to calculate ES score, use the built-in function:

```python
from mtist import infer_mtist as im

im.calculate_es_score(true_aij, inferred_aij)
```

In the above example, `true_aij` is the ground truthc community matrix used to generate the mtist dataset that an inference algorithm used to infer `inferred_aij`. Both `true_aij` and `inferred_aij` are numpy arrays.

### Benchmarking an algorithm using the MTIST package

#### Step 1: Create a Python function for your inference method
An easy way to infer and calculate ES score for each MTIST dataset is by using the tools available in the MTIST package. First, build a Python function that runs your inference algorithm with the following function signature:

```python
def my_inference_method(did, ...)

    ###
    # code to infer a SINGLE mtist dataset (load the data from disk, prepare the data, infer)
    ###

    return inferred_community_matrix
```

where `did` is a dataset ID (integer from 0 to 1,134) and the `inferred_community_matrix` is the inferred community matrix from that dataset ID.

Examples of the LinearRegression, ElasticNet, RidgeRegression, LassoRegression, and MKSeqSpike formatted in this manner can be found in the following locations:

| Inference Method                       | Location in Package                      |
|----------------------------------------|------------------------------------------|
| LinearRegression                       | infer_mtist.infer_from_did               |
| RidgeRegression (cross-validated)      | infer_mtist.infer_from_did_ridge_cv      |
| LassoRegression (cross-validated)      | infer_mtist.infer_from_did_lasso_cv      |
| ElasticNetRegression (cross-validated) | infer_mtist.infer_from_did_elasticnet_cv |
| MKSeqSpike (Rao et al. 2020)           | infer_mtist.infer_mkspikeseq_by_did      |

#### Step 2: Use the MTIST package to complete inference
To use the MTIST package to run your inference method over all of the MTIST datasets and calculate ES score for each, use the following code:

```python
im.INFERENCE_DEFAULTS.INFERENCE_PREFIX = "my_inference_method_name_"
im.INFERENCE_DEFAULTS.INFERENCE_FUNCTION = my_inference_method
_ = im.infer_and_score_all(save_inference=True, save_scores=True)
```
This code will use the `my_inference_method` function to infer each MTIST dataset, calculate ES score, and then save the ES score to disk in location `mtist1.0/mtist_datasets/my_inference_method_name_inference_result`. 

## Generating MTIST locally

In some cases, you'll want to generate  MTIST _in silico_ simulations on your own machine. This section describes that.

With default parameters:

```python

from mtist import master_dataset_generation as mdg
from mtist import assemble_mtist as am

mdg.generate_mtist_master_datasets()
am.assemble_mtist()

```

This requires (1) package installation and (2) `ground_truth` folder in your present working directory. For a full Python file describing this process, please see `mtist1.0/create_mtist_example.py`. 

### Altering default generation parameters

You can edit most conditions MTIST Generation uses to produce the datasets. For example,

```python
from mtist import master_dataset_generation as mdg
from mtist import assemble_mtist as am

from mtist import mtist_utils as mu

# Change the noise scale
mdg.MASTER_DATASET_DEFAULTS.NOISE_SCALES = [0.01, 0.05, 0.20] 

# Change the master dataset directory (MTIST numerical simulations before "sampling scheme" applied)
mu.GLOBALS.MASTER_DATASET_DIR = "my_new_directory" 

# Change the assembled MTIST dataset directory
mu.GLOBALS.MTIST_DATASET_DIR = "a_third_alternate_directory" 

----------------------------------------------
# Generate MTIST with these altered parameters
mdg.generate_mtist_master_datasets()
am.assemble_mtist()
```

Here is a table of default parameters one might want to change and their default value.

| Name                | Description                                                                                                           | Default value (type)                                            | Package location                                               |
|---------------------|-----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|----------------------------------------------------------------|
| MASTER_DATASET_DIR  | Relative path to directory with master datasets.                                                                      | "master_datasets" (str)                                         | mtist_utils.GLOBALS.MASTER_DATASET_DIR                         |
| MTIST_DATASET_DIR   | Relative path to directory with assembled MTIST datasets.                                                             | "mtist_datasets"  (str)                                         | mtist_utils.GLOBALS.MTIST_DATASET_DIR                          |
| GT_DIR              | Relative path to directory with ground truths.                                                                        | "ground_truths"   (str)                                         | mtist_utils.GLOBALS.GT_DIR                                     |
| random_seeds        | Random seeds used to generate up to 50 patients                                                                         | See further documentation for default value (list of length 50) | master_dataset_generation.MASTER_DATASET_DEFAULTS.random_seeds |
| noises              | Noise scales used in generation of master datasets                                                                    | [0.01, 0.05, 0.10] (list)                                       | master_dataset_generation.MASTER_DATASET_DEFAULTS.NOISE_SCALES       |
| INFERENCE_FUNCTION  | Function used to infer coefficient matrix from MTIST data. See further documentation to mimic its function signature. | Function handle                                                 | infer_mtist.INFERENCE_DEFAULTS.INFERENCE_FUNCTION              |



## Extended Information

Below are more lengthy explainations of a few concepts written above. (To-complete.)

### Altering the `INFERENCE_FUNCTION`
* Simply replace this with a new function handle with the following signature:
    * `new_function(did: int) -> ndarray`

Notice in this setup that the ndarray is the INFERRED Aij matrix of the corresponding ecosystem in the designated `did`.


### `random_seeds`
By default, these are the random seeds used for the fifty timeseries:

```python
    random_seeds = [   36656,  2369231,   416304, 10488077,  8982779, 12733201,
        9845126,  9036584,  5140131,  8493390,  3049039,  2753893,
       11563241,  5589942,  2091765,  2905119,  4240255, 10011807,
        5576645,   591973,  4211685,  9275155, 10793741,    41300,
        2858482,  6550368,  3346496, 12305126,  8717317,  6543552,
        5614865,  9104526, 10435541, 11942766,  6667140, 10471522,
         115475,  2721265,   309357,  9668522,  2698393,  9638443,
       11499954,  1444356,  8745245,  7964854,  1768742,  8139908,
       10646715, 10999907]
```

