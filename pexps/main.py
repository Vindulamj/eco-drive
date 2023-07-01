# MIT License

# Copyright (c) 2023 Vindula Jayawardana

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from imported_utils import *
from imported_utils.exp_utils import *
from env import *
from stats import SimulationStats
from scenarios.init_envs import NoStopNetwork

if __name__ == '__main__':
 
    c = NoStopNetwork.from_args(globals(), locals())
    stats = SimulationStats()
    # set the number of workers here
    n_workers = 12

    c.setdefaults( 
        sim_step=0.5, 
        adv_norm=False,
        batch_concat=True,
        render=False,
        step_save=5,
        use_ray=True,
        num_workers=n_workers,
        per_step_rollouts=n_workers, 

        max_accel=5.0,
        max_decel=-5.0,

        alg=TRPO,
        n_gds=10,
        n_minibatches=40,
        lr=1e-3,
        gamma=0.99,
        lam=0.97,
        opt='Adam',
        norm_reward=False,
        center_reward=False,
        use_critich=True,

        _n_obs=11,
        n_steps=3000,
        horizon=600,
        warmup_steps=50,
        act_type='accel',

        sumo_dir="sumo",
        # the model ID number used for testing
        e=0,
        test_run=False,
        wandb=False,
        wandb_dry_run=True,
        wandb_proj="eco-drive",
        wandb_key="9dbd690c152c02907b37359c88b9dbf4c9c6be0b",

        target_vel=15.0, 
        stats=stats,

        # every 'rl_fraction' vehicle is an RL vehicle. This means,
        # rl_fraction=1 => 100% RL vehicle penetration
        # rl_fraction=2 => 50% RL vehicle penetration
        # rl_fraction=5 => 20% RL vehicle penetration
        # rl_fraction=10 => 10% RL vehicle penetration
        rl_fraction=1,  

        # output files
        output_tracjectory_file=False,
    )

    # making sure the configurations are valid
    if c.test_run:
        c.use_ray = False
        c.num_workers = 1
        c.per_step_rollouts = 1
        c.render = False
        c.n_steps = 5
        c.wandb = False
        assert c.e > 0, "test run requires an existing model ID"
    else:
        c.use_ray = True
        c.num_workers = n_workers
        c.per_step_rollouts = n_workers
        c.render = False
        c.n_steps = 3000
        c.wandb = True
        c.wandb_dry_run = False
        c.wandb_proj = "eco-drive"

    c.run()