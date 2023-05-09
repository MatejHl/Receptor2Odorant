# Receptor2Odorant
Implementation of paper "**Matching receptor to odorant with protein language and graph neural networks**"
```
@inproceedings{
hladi{\v{s}}2023matching,
title={Matching receptor to odorant with protein language and graph neural networks},
author={Matej Hladi{\v{s}} and Maxence Lalis and Sebastien Fiorucci and J{\'e}r{\'e}mie Topin},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=q9VherQJd8_}
}
```

Data
----
Our aim is to keep the information about olfactory receptor experiments up-to-date, organized and easily accessible. We're going to keep all the data in one place at https://github.com/chemosim-lab/M2OR. We hope that it will be a useful resource for both biology practitioners as well as machine learning experts.

*Currently, we are working on a website for quick analyses of the data that everyone could use as a base resource. In the meantime the up-to-date data are available as a csv in the M2OR repo.*

To reporoduce the results in the paper, see **Original data** section below.

Usage
-----
For easy usage we created a Singularity container and included definition file in `_container/Receptor2Odorant_singularity.def`. You can find instructions how to bild the container [here](https://docs.sylabs.io/guides/3.0/user-guide/build_a_container.html#building-containers-from-singularity-definition-files). In further sections we assume the container was sucessfully built.
### Start the container in an interactive session:
```
singularity exec --containall --nv -B <bind_dir>:/mnt,<HOME>/.cache/huggingface/transformers Receptor2Odorant_singularity.sif bash
```
Source conda and activate the environment:
```
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate base
```
### Run scripts with a config file:
(Assuming that Receptor2Odorant dir is binded as /mnt)
```
python /mnt/Receptor2Odorant/scripts/main.py --config /mnt/Receptor2Odorant/configs/config_train.yml
```
or
```
python /mnt/Receptor2Odorant/scripts/main_odor_embedding.py --config /mnt/Receptor2Odorant/configs/config_odor_embedding_train.yml
```
There are several types of config files available:
`config_precompute.yml`,`config_train.yml`,`config_eval.yml`,`config_predict.yml`,`config_predict_single.yml`,`config_odor_embedding_train.yml`,`config_odor_embedding_predict.yml`
#### config_precompute:
This config file is giving instructions to precompute protein representation and saving it to .h5 file using `pyTables` library.
#### config_train:
The training config for the main OR-molecule model.
#### config_eval, config_predict, config_predict_single:
Evaluation and prediction configs. Difference between *predict* and *predict_single* is that the former expects a precomputed protein representation for a bulk of pairs and the second one is running protein representation model on the fly.
#### config_odor_embedding_train:
The config to train GNN-based odor prediction model that was used in section 6 *Agreement with odor perception* to create abstract odor families (i.e. clusters) for the molecules.
#### config_odor_embedding_predict:
Prediction config for odor prediction model which outputs molecule's embedding alongside its odor labels.


Original data
----
The link below leads to the original dataset used in the paper.

https://drive.google.com/drive/folders/1Sb6nPaSgeX66Wo5uG8NYRmcMfvGr76_K?usp=sharing

Download data file: `Data_figures.zip` and `Dataset.zip` unzip them and place in the giving folder (create the folders if necessary):
  - Content of `Data` from `Data_figures.zip` to `Receptor2Odorant/Figures/Data`
  - Content of `Dataset` from `Dataset.zip` to `Receptor2Odorant/RawData/m2or`
  - Content of `_seq_dist_matrices` from `_seq_dist_matrices.zip` to `Receptor2Odorant/RawData/m2or/_seq_dist_matrices`
