# Recursive Bayesian Networks
This repository contains the code to reproduce the results from the NeurIPS 2021 paper

Lieck R, Rohrmeier M (2021) **Recursive Bayesian Networks: Generalising and Unifying Probabilistic Context-Free Grammars and Dynamic Bayesian Networks**. In: _Proceedings of the 35th Conference on Neural Information Processing Systems (NeurIPS 2021)_

````
@inproceedings{lieck2021RBN,
  title = {Recursive {{Bayesian Networks}}: Generalising and {{Unifying Probabilistic Context}}-{{Free Grammars}} and {{Dynamic Bayesian Networks}}},
  booktitle = {Proceedings of the 35th {{Conference}} on {{Neural Information Processing Systems}} ({{NeurIPS}} 2021)},
  author = {Lieck, Robert and Rohrmeier, Martin},
  year = {2021},
}
````

## Installation

Download the code, create a fresh Python 3.9 environment and install all necessary dependencies via

````
$ pip install -r requirements.txt
````

We provide two separate branches:

- [NeurIPS_2021_with_data](https://github.com/robert-lieck/RBN/tree/NeurIPS_2021_with_data) contains the pretrained model for music and the evaluation results from the paper. Because of file size restrictions [music_pretrained.pt](./src/music_pretrained.pt.zip) is zipped, please unzip for use in [`Evaluation.ipynb`](./src/Evaluation.ipynb).
- [NeurIPS_2021_without_data](https://github.com/robert-lieck/RBN/tree/NeurIPS_2021_without_data) does not contain the pretrained model for music and the evaluation results from the paper (~120MB). If you do not want to download these data, please download the ZIP of this branch. Note that in that case, some things in [`Evaluation.ipynb`](./src/Evaluation.ipynb) will need to be slight adapted or do not work, because you will need to generate you own evaluation data and cannot plot results from the pretrained model for music.

## Results

To reproduce the results from the paper (incl. the relevant figures and the example from Appendix C), open [`Evaluation.ipynb`](./src/Evaluation.ipynb) and follow the instructions there.

