# predictive_activity

The goal of this repository is to re-analyse Demarchi et al. 2019 (Automatic and feature-specific prediction-related neural activity in the human auditory system) data to check if there is indeed predictive activity or not.
Data are available here: https://zenodo.org/record/3268713


# Instructions

1. Download the original data from ???

2. Setup a python environment with the required libraries.

3. Execute the python scripts `reproduce_and_reorder_sound.py` and `reproduce_and_reorder_omissions.py` to replicate the classifications performed in the original study + new ones.  
For each participant, you should get a new folder with two subfolders (one for sounds, another for omissions), each containing:  
- the confusion matrices of the classifier at each time point in `rd_to_rd_confmats.npz`  
- classification accuracies across the temporal generalization matrix in `*.npy*` files which are named `cv_<training condition>_(reord)_to_<testing condition>_(reord)_(sp)_scores.npy`

The suffix `reord` means that the training (or testing) data is random data epoched and reordered to reproduce the sequence of sounds found in the corresponding structured condition.  
The suffix `sp` stands for "**s**imple **p**rediction" and means that the classifier accuracy was calculated based on trying to decode the most likely stimulus instead of the presented one (see paper for more details).

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