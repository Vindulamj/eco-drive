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
from exp import *
from env import *
import math

# vehicle colors
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
RED = (255, 0, 0)

class NoStopRLEnv(Env):
    
    speeds_last_step = {}

    def reset(self):
        """
        Reset custom Env class .
        """

        # load the network and subscribe to state.
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
        for rl_id in rl_vehs:
            veh_edge[rl_id] = ts.get_edge(rl_id)
        # set actions for rl vehicles
        if not warmup:
            for i in range(len(rl_vehs)):
                car_id_num = int(rl_vehs[i].rsplit(".", 1)[-1])
                # for mixed traffic, use IDM controller for some vehicles
                if (car_id_num % c.rl_fraction) > 0.0001: 
                    # use IDM controller for acceleration
                    veh_edge[rl_vehs[i]] = ts.get_edge(rl_vehs[i])
                    ts.set_IDM_accel(rl_vehs[i], noise=False, variety=False)
                    ts.set_color(rl_vehs[i], CYAN)
                else:
                    # use learning-based controller for acceleration
                    act = action[i][0]
                    ts.accel(rl_vehs[i], act)
                    ts.set_color(rl_vehs[i], RED)
                
        super().step()
        # call this after step() function to get updated vehicle list
        vehicle_list = ts.get_vehicle_list()
        obs, ids = self.get_state(vehicle_list, c, ts)
        reward, _ = self.get_reward(vehicle_list, c, ts, rl_vehs, veh_edge)
        self.collect_stats(warmup, c, ts, vehicle_list)

        return Namespace(obs=obs, id=ids, reward=reward)

    def get_state(self, vehicle_list, c, ts):
        """
        state contains the following features.
            - ego-vehicle distance to intersection
            - ego vehicle speed
            - leader relative distance
            - leader speed
            - folower relative distance
            - follower speed
            - traffic light phase
            - traffic light time remaining in the current phase
            - phase the ego-vehicle belongs to.
        """
        obs = {}
        # "TL" is common to all vehicles 
        current_phase = ts.get_phase("TL")
        time_remaining = ts.remaining_phase_time("TL")
        # intersection program
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        for veh in vehicle_list:  

            if veh.startswith('rl'):
                car_id_num = int(veh.rsplit(".", 1)[-1])
                # observations are only collected for rl vehicles
                if car_id_num % c.rl_fraction != 0:  
                        continue

                # controlled rl vehicles are rendered in RED
                ts.set_color(veh, color=RED)

                tmp_obs = []
                # rl vehicle speed and distance to and from the intersection
                self_speed = ts.get_speed(veh)
                self_distance, self_lane = ts.get_dist_intersection(veh, 250)
                road_id = self_lane.split("_")[0]
                tmp_obs.append(self_distance/250) 
                tmp_obs.append(self_speed/c.target_vel)

                # leader speed and relative distance 
                # TODO limit the visible distance to d = 100m
                leader_info = ts.get_leader(veh)
                if leader_info is None:
                    if road_id not in incoming_roads:
                        leader_speed = c.target_vel # padding with maximum velocity
                        # TODO: change the max to be extracted from the network
                        leader_dist = 250 # padding with max
                    else:
                        # at this point we know that 'veh' is a leading vehicle in an incoming approach
                        leader_speed = c.target_vel # padding with maximum velocity
                        # TODO: change the max to be extracted from the network
                        leader_dist = 250 # padding with max
                else:
                    leader_id, leader_dist = leader_info
                    leader_speed = ts.get_speed(leader_id)
                tmp_obs.append(leader_speed/c.target_vel)
                # TODO: change the max to be extracted from the network
                tmp_obs.append(leader_dist/250)
                
                # follower speed and relative distance
                # TODO limit the visible distance to d = 100m
                follower_info = ts.get_follower(veh)
                if follower_info is None:
                    # fill with padding values
                    follower_speed = c.target_vel # padding with max
                    # TODO: change the max to be extracted from the network
                    follower_dist = 250 # padding with max
                else:
                    follower_id, follower_dist = follower_info
                    follower_speed = ts.get_speed(follower_id)
                tmp_obs.append(follower_speed/ c.target_vel)
                # TODO: change the max to be extracted from the network
                tmp_obs.append(follower_dist/250)

                if road_id not in incoming_roads:
                    tmp_obs.append(2) #padding
                    tmp_obs.append(2) #padding
                    tmp_obs.append(2) #padding
                else:
                    if current_phase == 0:
                        tmp_obs.append(1) # red light
                        tmp_obs.append(1) # next phase
                        tmp_obs.append((time_remaining + 4)/34)
                    elif current_phase == 2:
                        tmp_obs.append(1) # red light
                        tmp_obs.append(0) # next phase
                        tmp_obs.append((time_remaining + 4)/34)
                    elif current_phase == 1:
                        tmp_obs.append(0) # yellow light
                        tmp_obs.append(1) # next phase
                        tmp_obs.append(time_remaining/34)
                    elif current_phase == 3:
                        tmp_obs.append(0) # yellow light
                        tmp_obs.append(0) # next phase
                        tmp_obs.append(time_remaining/34)

                if road_id == "E2TL":
                    # phase_id_index = 1
                    tmp_obs.append(0) 
                    tmp_obs.append(0) 
                elif road_id == "N2TL":
                    # phase_id_index = 2
                    tmp_obs.append(0) 
                    tmp_obs.append(1) 
                elif road_id == "W2TL":
                    # phase_id_index = 1
                    tmp_obs.append(0) 
                    tmp_obs.append(0) 
                elif road_id == "S2TL":
                    # phase_id_index = 2
                    tmp_obs.append(0) 
                    tmp_obs.append(1) 
                elif road_id.startswith( ':TL' ):
                    # middle of the intersection 
                    # phase_id_index = 3
                    tmp_obs.append(1) 
                    tmp_obs.append(0) 
                else:
                    # in an outgoing lane
                    # phase_id_index = 4
                    tmp_obs.append(1) 
                    tmp_obs.append(1) 
                obs[veh] = np.array(tmp_obs)

        sort_id = lambda d: [v for k, v in sorted(d.items())]
        ids = sorted(obs)
        obs = arrayf(sort_id(obs)).reshape(-1, c._n_obs)
        return obs, ids

    def fuel_model(self, v_speed, v_accel):
        """
        VT-CPFM Fuel Model.
        """
        R_f = 1.23*0.6*0.98*3.28*(v_speed**2) + 9.8066*3152*(1.75/1000)*0.033*v_speed + 9.8066*3152*(1.75/1000)*0.033 + 9.8066*3152*0
        power = ((R_f + 1.04*3152*v_accel)/(3600*0.92)) * v_speed
        fuel = 0
        if power >= 0:
            fuel = 0.00078 + 0.000019556*power + 0.000001*(power**2)
        else:
            fuel = 0.00078
        return fuel


    def get_reward(self, vehicle_list, c, ts, old_vehicle_list, veh_edge):
        """
        compute the reward of the previous action.
        """
        max_speed = c.target_vel
        
        horizontal_speeds = []
        vertical_speeds = []
        horizontal_fuels = []
        vertical_fuels = []

        for veh in vehicle_list:
            v_speed = ts.get_speed(veh)
            v_accel = ts.get_acceleration(veh)
            v_edge = ts.get_edge(veh)

            if v_edge.startswith("E2TL") or v_edge.startswith("TL2W"):
                horizontal_speeds.append(v_speed)
            else:
                vertical_speeds.append(v_speed)

            fuel = self.fuel_model(v_speed, v_accel)

            if v_edge.startswith("E2TL") or v_edge.startswith("TL2W"):
                horizontal_fuels.append(fuel)
            else:
                vertical_fuels.append(fuel)

        # stop reward
        horizontal_stops = 0
        vertical_stops = 0
        h_vehs = 0
        v_vehs = 0
        for veh in vehicle_list:
            tmp_speed = ts.get_speed(veh)
            tmp_edge = ts.get_edge(veh)
            if tmp_edge.startswith("E2TL") or tmp_edge.startswith("TL2W"):
                h_vehs += 1
            else:
                v_vehs += 1
            if tmp_speed < 0.3:
                if tmp_edge.startswith("E2TL") or tmp_edge.startswith("TL2W"):
                    horizontal_stops += 1
                else:
                    vertical_stops += 1 

        slow_v_veh = False
        slow_h_veh = False 
        for veh in vehicle_list:
            v_edge = ts.get_edge(veh)
            if v_edge.startswith("E2TL") :
                v_pos = ts.get_position(veh)
                v_speed = ts.get_speed(veh)
                if v_pos < 50:
                    if v_speed < 5 and not slow_h_veh:
                        slow_h_veh = True
            elif v_edge.startswith("S2TL") :
                v_pos = ts.get_position(veh)
                v_speed = ts.get_speed(veh)
                if v_pos < 50:
                    if v_speed < 5 and not slow_v_veh:
                        slow_v_veh = True

        lead_h_speed = 0
        lead_v_speed = 0
        for veh in vehicle_list:
            leader_info = ts.get_leader(veh)
            v_speed = ts.get_speed(veh)
            v_edge = ts.get_edge(veh)
            if leader_info == None:
                if v_edge.startswith("S2TL") and lead_v_speed == 0:
                    lead_v_speed = v_speed
                elif v_edge.startswith("E2TL") and lead_h_speed == 0:
                    lead_h_speed = v_speed

        stop_h_reward = 0
        stop_v_reward = 0
        if h_vehs != 0:
            stop_h_reward = -1*(horizontal_stops)/h_vehs
        if v_vehs != 0:
            stop_v_reward = -1*(vertical_stops)/v_vehs
            
        avg_h_speed_reward = np.mean(np.array(horizontal_speeds)/max_speed)
        avg_v_speed_reward = np.mean(np.array(vertical_speeds)/max_speed)

        avg_h_fuel_reward = (np.mean(np.array(horizontal_fuels)))
        avg_v_fuel_reward = (np.mean(np.array(vertical_fuels)))

        if slow_h_veh:
            tot_h_reward = -100
        elif avg_h_fuel_reward <= 0.00078 and horizontal_stops == 0:
            tot_h_reward = 5*(math.exp(avg_h_speed_reward) - 1)
        elif avg_h_fuel_reward <= 0.00078 and horizontal_stops > 0:
            tot_h_reward = 5*(math.exp(avg_h_speed_reward) - 1) + 10*stop_h_reward
        else:
            tot_h_reward = -3*(math.exp(1000*avg_h_fuel_reward)-1) + 4*(math.exp(avg_h_speed_reward) - 1) + 10*stop_h_reward

        if slow_v_veh:
            tot_v_reward = -100
        elif avg_v_fuel_reward <= 0.00078 and vertical_stops == 0:
            tot_v_reward = 5*(math.exp(avg_v_speed_reward) - 1)
        elif avg_v_fuel_reward <= 0.00078 and vertical_stops > 0:
            tot_v_reward = 5*(math.exp(avg_v_speed_reward) - 1) + 10*stop_v_reward
        else:
            tot_v_reward = -3*(math.exp(1000*avg_v_fuel_reward)-1) + 4*(math.exp(avg_v_speed_reward) - 1) + 10*stop_v_reward

        rewards = {}
        
        for rl_id in old_vehicle_list:
            tmp_edge = veh_edge[rl_id]
            if tmp_edge.startswith("E2TL") or tmp_edge.startswith("TL2W"):
                rewards[rl_id] = tot_h_reward/h_vehs
            else:
                rewards[rl_id] = tot_v_reward/v_vehs

        sort_id = lambda d: [v for k, v in sorted(d.items())]
        ids = sorted(rewards)
        rewards = arrayf(sort_id(rewards)).reshape(-1, 1)
        return rewards, ids


    def collect_stats(self, warmup, c, ts, vehicle_list):
        # collect stats of the current step

        for veh in vehicle_list:
            v_accel = ts.get_acceleration(veh)
            v_speed = ts.get_speed(veh)
            v_lane = ts.get_lane_id(veh)
            v_lane_pos = ts.get_position(veh)
            v_fuel = self.fuel_model(v_speed, v_accel)
            v_emiission = ts.get_co2_emission(veh)*c.sim_step

            if warmup:
                if v_lane.startswith('TL2'):
                    # TODO remove this hard coded values
                    data_list = [v_speed, v_fuel, v_emiission, 0, 1] if v_lane_pos > 240 else [v_speed, v_fuel, v_emiission, 0, 0]
                else:
                    data_list = [v_speed, v_fuel, v_emiission, 0, 0]
            else:
                if v_lane.startswith('TL2'):
                    # TODO remove this hard coded values
                    data_list = [v_speed, v_fuel, v_emiission, 1, 1] if v_lane_pos > 240 else [v_speed, v_fuel, v_emiission, 1, 0]
                else:
                    data_list = [v_speed, v_fuel, v_emiission, 1, 0]

            if veh not in c.stats.veh_data.keys():
                c.stats.veh_data[veh] = [data_list]
            else:
                c.stats.veh_data[veh].append(data_list)