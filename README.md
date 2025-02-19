## Table of Contents
- [About](#-about)
- [How to use](#-how-to-use)
- [License](#-license)

## 🚀 About

The goal of this repository is to re-analyse Demarchi et al. 2019 (Automatic and feature-specific prediction-related neural activity in the human auditory system) data and reproduce their results. We conclude using two approaches (an empirical one and a theoretical one) that there is no evidence of feature-specific prediction-related neural activity in this dataset. We thank Demarchi et al. for making their data available through open access: https://zenodo.org/record/3268713

# How to use

1. Download the original data from Demarchi et al. (2019) at this link: https://zenodo.org/record/3268713

2. Clone the repository and setup a python environment with the required libraries in the `requirements.txt`.

Using [astral-uv](https://docs.astral.sh/uv/getting-started/) for example, you can:

    ```bash
    git clone https://github.com/romquentin/predictive_activity.git
    uv venv --python 3.9
    source .venv/bin/activate
    uv pip install -r requirements.txt
    ```

3. Execute the python scripts `reproduce_and_reorder_sound.py` and `reproduce_and_reorder_omissions.py` to replicate the decoding performed in the original Demarchi et al. study and our empirical approach that invalidate their results. Please note that you will need to adjust the paths (line 16 and 17) to access the data according to where you have downloaded it. You also have to add an argument `--subject` with the subject number (e.g., 0). For each participant, you should get a new folder with two subfolders (one for sounds, another for omissions), each containing:  

- the confusion matrices of the classifier at each time point in `rd_to_rd_confmats.npz`  
- classification accuracies across the temporal generalization matrix in `*.npy*` files which are named `cv_<training condition>_(reord)_to_<testing condition>_(reord)_(sp)_scores.npy`

The suffix `reord` means that the training (or testing) data is random data epoched and reordered to reproduce the sequence of sounds found in the corresponding structured condition.  
The suffix `sp` stands for "**s**imple **p**rediction" and means that the classifier accuracy was calculated based on trying to decode the most likely stimulus instead of the presented one (see the manuscript for more details).
The `reproduce_and_reorder_sound.py` takes approximately one hour per participant. The `reproduce_and_reorder_omission.py` is several times faster.

A sample of output data containing all classification results for 2 participants is provided in the `/sample` folder.

4. Setup a R environment with the required libraries, as well as an IDE which is able to open and run Quarto documents.  
Required libraries:  
- expm=0.999.7  
- patchwork=1.2.0  
- magrittr=2.0.3  
- tictoc=1.2  
- correlation=0.8.5  
- abind=1.4.5  
- tidyverse=2.0.0  
- reticulate=1.34.0  

5. Go to `postprocessing.qmd` and set the root folder where the output data from step 3 should be found.

6. Execute `postprocessing.qmd` to build all the figures and supplementary figures of the paper, including the calculation of theoretical accuracy and cluster-based permutation statistical tests.  
Total execution time on all participants ~30min on a regular laptop.

## 📃 License

BSD 3-Clause License