# Learning Eco-Driving Strategies at Signalized Intersections

This repo contains the codebase for the paper titled "Learning Eco-Driving Strategies at Signalized Intersections" published in European Control Conference (ECC) 2022.

[Paper](https://arxiv.org/pdf/2204.12561.pdf) | [Website](https://vindulamj.github.io/eco-driving-rl/)

## Installation
1. The code has been tested on Ubuntu 20.04, or MacOS 13.0 Ventura.
2. Follow instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to install Miniconda, likely `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh` followed by `bash Miniconda3-latest-Linux-x86_64.sh`.
3. Follow instructions in SUMO [website](https://eclipse.dev/sumo/) to install SUMO simulator. Try running `sumo` in terminal to confirm the successful installation.
4. Install PyTorch from [pytorch.org](pytorch.org).
5. Install dependencies `pip install -r requirements.txt`.

## Instructions
`<agent_type>` is the type of the agents that can be used to control CAVs. Available options: RL, IDM, ECO_CACC

`<res_dir>` is the result directory, which is where the model checkpoints, training logs, and training csv results will be saved. Add `render` as an argument for using `sumo-gui` instead of `sumo`. E.g. `python pexps/<script>.py <res_dir> render`.

### Eco-driving in 1x1 intersection
`python pexps/main.py --agent <agent_type> --res <res_dir>`  

## Citation 

If you are using this codebase for any purpose please cite the following paper. 

```
@INPROCEEDINGS{9838000,
  author={Jayawardana, Vindula and Wu, Cathy},
  booktitle={2022 European Control Conference (ECC)}, 
  title={Learning Eco-Driving Strategies at Signalized Intersections}, 
  year={2022},
  pages={383-390},
  doi={10.23919/ECC55457.2022.9838000}}
```