import imp
from numpy.lib.polynomial import roots
from imported_utils import *
from exp import *
from env import *
from .no_stop import NoStopRLEnv
from .no_stop_idm import NoStopIDMEnv
from .no_stop_eco_cacc import NoStopEcoCACCEnv
from .no_stop_baseline import NoStopBaselineEnv
from .no_stop2x2 import NoStopRL2x2Env

class NoStopNetwork(Main):

    def create_env(c):
        c._norm = NormEnv(c, None)
        print(c.agent)
        if c.agent == "RL":
            return NoStopRLEnv(c)
        elif c.agent == "IDM":
            return NoStopIDMEnv(c)
        elif c.agent == "ECO-CACC":
            return NoStopEcoCACCEnv(c)
        elif c.agent == "RL-2x2":
            return NoStopRL2x2Env(c)
        elif c.agent == "BASELINE":
            return NoStopBaselineEnv(c)
        else:
            raise Exception("Invalid agent type!")
            
    @property
    def observation_space(c):
        low = np.full(c._n_obs, -1)
        return Box(low, np.ones_like(low))

    @property
    def action_space(c):
        if c.act_type == 'accel':
            return Box(low=c.max_decel, high=c.max_accel, shape=(1,), dtype=np.float32)
        else:
            return Discrete(c.n_actions)