# No-stop Intersections

## Installation
1. Make sure that your computer's or server's OS is Ubuntu 18.04 or lower, or Mac.
2. Follow instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to install Miniconda, likely `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh` followed by `bash Miniconda3-latest-Linux-x86_64.sh`
3. Run `bash scripts/setup_sumo_<os_version>.sh` corresponding to your OS version to set up SUMO and add `~/sumo_binaries/bin` to your `PATH` environment variable. Try running `sumo`
4. Install PyTorch from [pytorch.org](pytorch.org).
5. Install dependencies `pip install -r requirements.txt`

## Run Instructions
`<agent_type>` is the type of the agents that can be used to control CAVs. Available options: RL, IDM, ECO_CACC

`<res_dir>` is the result directory, which is where the model checkpoints, training logs, and training csv results will be saved. Add `render` as an argument for using `sumo-gui` instead of `sumo`. E.g. `python pexps/<script>.py <res_dir> render`.

### Eco-driving in 1x1 intersection
`python pexps/main.py --agent <agent_type> --res <res_dir>`

This code base was adapted from Zhongxia Yan's FlowLite repository.  
