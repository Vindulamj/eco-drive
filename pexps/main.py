from imported_utils import *
from imported_utils.exp_utils import *
from env import *
from scenarios.init_envs import NoStopNetwork

if __name__ == '__main__':
 
    c = NoStopNetwork.from_args(globals(), locals())
    # set the number of workers here
    n_workers = 1
    c.setdefaults( 
 
        sim_step=0.5, 
        adv_norm=False,
        batch_concat=True,
        render=True,
        step_save=5,
        use_ray=False,
        num_workers=n_workers,
        per_step_rollouts=n_workers*1, 

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

        _n_obs=11,
        n_steps=5,
        horizon=600,
        warmup_steps=50,
        act_type='accel',

        sumo_dir="sumo",
        e=2750,
        test_run=True,
        wandb=False,
        wandb_dry_run=True,
        wandb_proj="no-stop-intersections",

        target_vel=15.0, 
        avg_reward=0,
        running_steps=0,
        velocity_fleet=0,
        rl_speed=0,
        fuel_fleet=0,
        veh_data={},
        veh_fuel_data_avg=[],
        veh_emission_data_avg=[],
        veh_speed_data_avg=[],

        rl_fraction=1, # every 'rl_fraction' vehicle is an RL vehicle   4: 75%, 2: 50%, 4/3: 25%
        enable_mixed_traffic=True
    )
    if c.test_run:
        c.use_ray = False
        print("Override \"use_ray\" variable if in test mode")

    if c.alg == "PPO" or c.alg == "TRPO":
        c.use_critic = True

    # run experiment
    c.run()