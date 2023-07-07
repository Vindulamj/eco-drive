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

from imported_utils import *
from exp import *
from env import *
from .no_stop import NoStopRLEnv
from .no_stop_idm import NoStopIDMEnv
from .no_stop_baseline import NoStopBaselineEnv


class NoStopNetwork(Main):
    def create_env(c):
        c._norm = NormEnv(c, None)
        print(c.agent)
        if c.agent == "RL":
            return NoStopRLEnv(c)
        elif c.agent == "IDM":
            return NoStopIDMEnv(c)
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
        if c.act_type == "accel":
            return Box(low=c.max_decel, high=c.max_accel, shape=(1,), dtype=np.float32)
        else:
            return Discrete(c.n_actions)
