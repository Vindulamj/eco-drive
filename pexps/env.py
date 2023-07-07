# MIT License

# Copyright (c) 2023 Vindula Jayawardana and Zhongxia Yan

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


import gym

from ut import *
from sumo_utils.general import *
from sumo_utils.xml import *
from sumo_utils.sumo import *


class Env:
    """
    Offers a similar reinforcement learning environment interface as gym.Env
    Wraps around a TrafficState (ts) and the SUMO traci (tc)
    """

    def __init__(self, c):
        self.c = c.setdefaults(
            redef_sumo=False,
            warmup_steps=0,
            skip_stat_steps=0,
            skip_vehicle_info_stat_steps=True,
        )
        self.sumo_def = SumoDef(c)
        self.tc = None
        self.ts = None
        self.rollout_info = NamedArrays()
        self._vehicle_info = [] if c.get("vehicle_info_save") else None
        self._step = 0

    def def_sumo(self, *args, **kwargs):
        """Override this with code defining the SUMO network"""
        # https://sumo.dlr.de/docs/Networks/PlainXML.html
        # https://sumo.dlr.de/docs/Definition_of_Vehicles,_Vehicle_Types,_and_Routes.html
        return self.sumo_def.save(*args, **kwargs)

    def step(self, *args):
        """
        Override this with additional code which applies acceleration and measures observation and reward
        """
        c = self.c
        self.ts.step()
        self._step += 1
        return c.observation_space.low, 0, False, None

    def reset_sumo(self):
        """
        Loads the sumo network.
        """
        c = self.c
        sumo_def = self.sumo_def

        generate_def = c.redef_sumo or not sumo_def.sumo_cmd
        if generate_def:
            kwargs = {}
            kwargs["net"] = sumo_def.return_sumo_net(**kwargs)
            sumo_def.sumo_cmd = sumo_def.generate_sumo(**kwargs)
        self.tc = sumo_def.start_sumo(self.tc)

        if generate_def:
            self.sumo_paths = {
                k: p for k, p in kwargs.items() if k in SumoDef.file_args
            }
            defs = {k: E.from_path(p) for k, p in self.sumo_paths.items()}
            self.ts = TrafficState(c, self.tc, **defs)
        else:
            self.ts.reset(self.tc)
        self.ts.setup()
        return True

    def init_env(self):
        self.rollout_info = NamedArrays()
        self._step = 0
        # IMPORTANT: one step in the simulation at the begining.
        ret = self.step(warmup=True)
        i = 0
        for i in range(self.c.warmup_steps):
            ret = self.step(warmup=True)
        print("Warmed up for " + str(i + 1) + " steps")

        if isinstance(ret, tuple):
            return ret[0]
        return {k: v for k, v in ret.items() if k in ["obs", "id"]}

    def reset(self):
        while not self.reset_sumo():
            pass
        return self.init_env()

    def close(self):
        try:
            traci.close()
        except:
            pass


class NormEnv(gym.Env):
    """
    Reward normalization with running average https://github.com/joschu/modular_rl
    """

    def __init__(self, c, env):
        self.c = c.setdefaults(
            norm_obs=False,
            norm_reward=False,
            center_reward=False,
            reward_clip=np.inf,
            obs_clip=np.inf,
        )
        self.env = env
        self._obs_stats = RunningStats()
        self._return_stats = RunningStats()
        self._reward_stats = RunningStats()
        self._running_ret = 0

    def norm_obs(self, obs):
        c = self.c
        if c.norm_obs:
            self._obs_stats.update(obs)
            obs = (obs - self._obs_stats.mean) / (self._obs_stats.std + 1e-8)
            obs = np.clip(obs, -c.obs_clip, c.obs_clip)
        return obs

    def norm_reward(self, reward):
        c = self.c
        if c.center_reward:
            self._reward_stats.update(reward)
            reward = reward - self._reward_stats.mean
        if c.norm_reward:
            self._running_ret = (
                self._running_ret * c.gamma + reward
            )  # estimation of return
            self._return_stats.update(self._running_ret)
            normed_r = reward / (
                self._return_stats.std + 1e-8
            )  # norm reward by std of return
            reward = np.clip(normed_r, -c.reward_clip, c.reward_clip)
        return reward

    def reset(self):
        return self.norm_obs(self.env.reset())

    def step(self, action=None):
        ret = self.env.step(action)
        if isinstance(ret, tuple):
            obs, reward, done, info = ret
            norm_obs = self.norm_obs(obs)
            norm_rew = self.norm_reward(reward)
            return norm_obs, norm_rew, done, reward
        else:
            if "obs" in ret:
                ret["obs"] = self.norm_obs(ret["obs"])
            ret["raw_reward"] = ret["reward"]
            ret["reward"] = self.norm_reward(ret["reward"])
            return ret

    def __getattr__(self, attr):
        try:
            return self.__getattribute__(attr)
        except:
            return getattr(self.env, attr)
