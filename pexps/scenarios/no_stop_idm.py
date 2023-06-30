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


from numpy.lib.polynomial import roots
from imported_utils import *
from exp import *
from env import *

# Vehicle colors
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
RED = (255, 0, 0)

class NoStopIDMEnv(Env):
    
    speeds_last_step = {}

    def reset(self):
        """
        Custom Env class reset method. 
        """
        # load the network and subscribe to state 
        while not self.reset_sumo():
            pass
        ret = super().init_env()
        c = self.c
        ts = self.ts
        c.lane_length = ts.get_lane_length("S2TL_0")
        c.intersection_length = ts.get_lane_length(":TL_1_0")
        return ret

    def step(self, action=[], rl_vehs=[], t=0, warmup=False):
        """
        Build observations, rewards and also simulate the environment by one step.
        """
        c = self.c
        ts = self.ts
        veh_edge = {}
        # collect roads before doing an env step
        if not warmup:
            for rl_id in rl_vehs:
                veh_edge[rl_id] = ts.get_edge(rl_id)
                ts.set_IDM_accel(rl_id, noise=False, variety=False)

        super().step()
        # call this after step() function to get updated vehicle list
        vehicle_list = ts.get_vehicle_list()
        obs, ids = self.get_state(vehicle_list, c, ts)
        reward, _ = self.get_reward(vehicle_list, c, ts, rl_vehs, veh_edge)
        self.collect_stats(warmup, c, ts, vehicle_list, reward)

        return Namespace(obs=obs, id=ids, reward=reward)

    def get_state(self, vehicle_list, c, ts):
        """ Placeholder state information"""
        obs = {}

        for veh in vehicle_list:  

            if veh.startswith('rl'):
                obs[veh] = np.zeros(c._n_obs)

        sort_id = lambda d: [v for k, v in sorted(d.items())]
        ids = sorted(obs)
        obs = arrayf(sort_id(obs)).reshape(-1, c._n_obs)
        return obs, ids

    def fuel_model(self, v_speed, v_accel):
        """VT-CPFM Fuel Model"""
        R_f = 1.23*0.6*0.98*3.28*(v_speed**2) + 9.8066*3152*(1.75/1000)*0.033*v_speed + 9.8066*3152*(1.75/1000)*0.033 + 9.8066*3152*0
        power = ((R_f + 1.04*3152*v_accel)/(3600*0.92)) * v_speed
        fuel = 0
        if power >= 0:
            fuel = 0.00078 + 0.000019556*power + 0.000001*(power**2)
        else:
            fuel = 0.00078
        return fuel


    def get_reward(self, vehicle_list, c, ts, old_vehicle_list, veh_edge):
        """Compute the reward of the previous action."""

        rewards = {}
        
        for rl_id in old_vehicle_list:
            rewards[rl_id] = 0

        sort_id = lambda d: [v for k, v in sorted(d.items())]
        ids = sorted(rewards)
        rewards = arrayf(sort_id(rewards)).reshape(-1, 1)
        return rewards, ids


    def collect_stats(self, warmup, c, ts, vehicle_list, reward):
        # collect stats
        all_speeds = []
        rl_speeds = [] 
        all_accels = []
        rl_accels = []
        all_fuel = []

        for veh in vehicle_list:
            v_accel = ts.get_acceleration(veh)
            v_speed = ts.get_speed(veh)
            v_lane = ts.get_lane_id(veh)
            v_lane_pos = ts.get_position(veh)
            v_fuel = self.fuel_model(v_speed, v_accel)
            v_emiission = ts.get_co2_emission(veh)*c.sim_step
            if veh.startswith('rl'):
                rl_speeds.append(v_speed)
                rl_accels.append(v_accel)
            all_speeds.append(v_speed)
            all_accels.append(v_accel)
            all_fuel.append(v_fuel)

            if warmup:
                if v_lane.startswith('TL2'):
                    if v_lane_pos > 240:
                        data_list = [v_speed, v_fuel, v_emiission, 0, 1]
                    else:
                        data_list = [v_speed, v_fuel, v_emiission, 0, 0]
                else:
                    data_list = [v_speed, v_fuel, v_emiission, 0, 0]
            else:
                if v_lane.startswith('TL2'):
                    if v_lane_pos > 240:
                        data_list = [v_speed, v_fuel, v_emiission, 1, 1]
                    else:
                        data_list = [v_speed, v_fuel, v_emiission, 1, 0]
                else:
                    data_list = [v_speed, v_fuel, v_emiission, 1, 0]

            if veh not in c.veh_data.keys():
                c.veh_data[veh] = [data_list]
            else:
                c.veh_data[veh].append(data_list)

        if not warmup:
            c.avg_reward += reward[0][0]
            c.running_steps += 1

            if len(all_speeds) == 0:
                avg_speed = 0
            else:
                avg_speed = np.mean(np.array(all_speeds))
            
            if len(all_fuel) == 0:
                sum_fuel = 0
            else:
                sum_fuel = np.sum(np.array(all_fuel))

            if len(rl_speeds) == 0:
                avg_rl_speed = 0
            else:
                avg_rl_speed = np.mean(np.array(rl_speeds))

            c.velocity_fleet += avg_speed
            c.rl_speed += avg_rl_speed
            c.fuel_fleet += sum_fuel

        