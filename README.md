# The MTIST Platform
_**m**icrobiome **ti**me **s**eries **t**est standard dataset_

MTIST is a standardized test dataset designed to benchmark microbial ecosystem inference algorithms. In this repository, we provide both the code used to generate MTIST _and_ instructions of how to benchmark an algorithm using MTIST. 

TO-DO: Separate out this readme into introduction/user guide/etc.

# Installation
Install with `pip` in "editable" mode.

1. Clone repo
2. Navigate to folder
3. Create virtual environment
4. Install in editable mode using `pip install -e .`

Project will soon be uploaded to PyPi and BioConda.


# Usage

## Benchmarking an inference algorithm

To test the performance of a microbial ecosystem inference algorithm, run your inference algorithm over each dataset in MTIST. Here are the locations of the required data for benchmarking:

| Name                      | Description                                                                                                         | Location                                     |
|---------------------------|---------------------------------------------------------------------------------------------------------------------|----------------------------------------------|
| Individual MTIST datasets | All 1,134 _in silico_ time series to be used for inference                                                          | `mtist1.0/mtist_datasets/dataset_*.csv`      |
| MTIST dataset metadata    | Describes the conditions used to generate each MTIST dataset                                                        | `mtist1.0/mtist_datasets/mtist_metadata.csv` |
| Ground truths             | The underlying maximum growth rates and interaction coefficients used to generate each MTIST _in silico_ simulation | `mtist1.0/ground_truths/*`                   |



## Generating MTIST locally

In some cases, you'll want to generate the MTIST _in silico_ simulations on your own machine. This section describes that.

With default parameters:

```python

from mtist import master_dataset_generation as mdg
from mtist import assemble_mtist as am

mdg.generate_mtist_master_datasets()
am.assemble_mtist()

```

This requires (1) package installation and (2) `ground_truth` folder in your present working directory.

### Altering default generation parameters

You can edit most conditions MTIST Generation uses to produce the datasets. For example,

```python
from mtist import master_dataset_generation as mdg
mdg.MASTER_DATASET_DEFAULTS.NOISE_SCALES = [0.01, 0.05, 0.20] # if you wanted to see what different noise scale parameters would look like

import mtist_utils
mtist_utils.GLOBALS.MASTER_DATASET_DIR = "my_new_directory" # change where the master datasets go
mtist_utils.GLOBALS.MTIST_DATASET_DIR = "a_third_alternate_directory" # change where assembled MTIST datasets get saved/where infernece results are saved, etc.
```

Here is a table of default parameters one might want to change and the value that must be edited to change it.

| Name                | Description                                                                                                           | Default value (type)                                            | Package location                                               |
|---------------------|-----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|----------------------------------------------------------------|
| random_seeds        | Random seeds used to generate the 50 patients                                                                         | See further documentation for default value (list of length 50) | master_dataset_generation.MASTER_DATASET_DEFAULTS.random_seeds |
| noises              | Noise scales used in generation of master datasets                                                                    | [0.01, 0.05, 0.10] (list)                                       | master_dataset_generation.MASTER_DATASET_DEFAULTS.NOISE_SCALES       |
| seq_depth_th        | Relative abundance threshold for implementing simulated sequencing depth (see paper Methods)                          | 0.1 (float)                                                     | assemble_mtist.ASSEMBLE_MTIST_DEFAULTS.seq_depth_th            |
| INFERENCE_FUNCTION  | Function used to infer coefficient matrix from MTIST data. See further documentation to mimic its function signature. | Function handle                                                 | infer_mtist.INFERENCE_DEFAULTS.INFERENCE_FUNCTION              |
| inference_threshold | Threshold used to floor the floored inference result                                                                  | 1/3 (float)                                                     | infer_mtist.INFERENCE_DEFAULTS.inference_threshold             |
| MASTER_DATASET_DIR  | Relative path to directory with master datasets.                                                                      | "master_datasets" (str)                                         | mtist_utils.GLOBALS.MASTER_DATASET_DIR                         |
| GT_DIR              | Relative path to directory with ground truths.                                                                        | "ground_truths"   (str)                                         | mtist_utils.GLOBALS.GT_DIR                                     |
| MTIST_DATASET_DIR   | Relative path to directory with assembled MTIST datasets.                                                             | "mtist_datasets"  (str)                                         | mtist_utils.GLOBALS.MTIST_DATASET_DIR                          |


## Benchmarking using MTIST

To best use MTIST, apply your inference algorithm to each dataset ID, and organize your inference results like so:

(WIP)



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

