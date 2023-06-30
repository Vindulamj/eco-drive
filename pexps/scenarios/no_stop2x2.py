from numpy.lib.polynomial import roots
from imported_utils import *
from exp import *
from env import *
import math

# Vehicle colors
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
RED = (255, 0, 0)

class NoStopRL2x2Env(Env):
    
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
        for rl_id in rl_vehs:
            veh_edge[rl_id] = ts.get_edge(rl_id)
        # set actions for rl vehicles
        if not warmup:
            for i in range(len(rl_vehs)):
                act = action[i][0]
                ts.accel(rl_vehs[i], act)

        super().step()
        # call this after step() function to get updated vehicle list
        vehicle_list = ts.get_vehicle_list()
        obs, ids = self.get_state(vehicle_list, c, ts)
        reward, _ = self.get_reward(vehicle_list, c, ts, rl_vehs, veh_edge)
        self.collect_stats(warmup, c, ts, vehicle_list, reward)

        return Namespace(obs=obs, id=ids, reward=reward)

    def get_state(self, vehicle_list, c, ts):
        """
        State contains the following features.
            - ego-vehicle distance to intersection
            - ego vehicle speed
            - leader relative distance
            - leader speed
            - folower relative distance
            - follower speed
            - traffic light phase
            - traffic light time remaining in the current phase
            - phase the ego-vehicle belongs to 
        """
        obs = {}
        # intersection program
        incoming_roads = ["E2TL1","E2TL2","E2TL3","E2TL4", "N2TL1","N2TL2","N2TL3","N2TL4", "W2TL", "W2TL1", "W2TL2", "W2TL3", "W2TL4", "S2TL1","S2TL2","S2TL3","S2TL4"]
        for veh in vehicle_list:  

            if veh.startswith('rl'):
                car_id_num = int(veh.rsplit(".", 1)[-1])
                if car_id_num % c.rl_fraction != 0:  # for mixed traffic
                    continue

                tmp_obs = []
                # rl vehicle speed and distance to and from the intersection
                self_speed = ts.get_speed(veh)
                self_distance, self_lane = ts.get_dist_intersection(veh, 250)
                road_id = self_lane.split("_")[0]
                tmp_obs.append(self_distance/250) # rl position
                tmp_obs.append(self_speed/c.target_vel)

                # leader speed and relative distance 
                # TODO limit the visible distance to d = 100m
                leader_info = ts.get_leader(veh)
                if leader_info == None:
                    if road_id not in incoming_roads:
                        leader_speed = c.target_vel # padding with maximum velocity
                        leader_dist = 250 # padding with max
                    else:
                        # at this point we know that 'veh' is a leading vehicle in an incoming approach
                        leader_speed = c.target_vel # padding with maximum velocity
                        leader_dist = 250 # padding with max
                else:
                    leader_id, leader_dist = leader_info
                    leader_speed = ts.get_speed(leader_id)
                tmp_obs.append(leader_speed/c.target_vel)
                tmp_obs.append(leader_dist/250)
                
                # follower speed and relative distance
                # TODO limit the visible distance to d = 100m
                follower_info = ts.get_follower(veh)
                if follower_info == None:
                    # fill with padding values
                    follower_speed = c.target_vel # padding with max
                    follower_dist = 250 # padding with max
                else:
                    follower_id, follower_dist = follower_info
                    follower_speed = ts.get_speed(follower_id)
                tmp_obs.append(follower_speed/ c.target_vel)
                tmp_obs.append(follower_dist/250)

                if road_id == "E2TL1" or road_id == "S2TL1" or road_id == "N2TL1" or road_id == "W2TL1":
                    current_phase = ts.get_phase("B0")
                    time_remaining = ts.remaining_phase_time("B0")
                elif road_id == "E2TL2" or road_id == "S2TL2" or road_id == "N2TL2" or road_id == "W2TL2":
                    current_phase = ts.get_phase("A0")
                    time_remaining = ts.remaining_phase_time("A0")
                elif road_id == "E2TL3" or road_id == "S2TL3" or road_id == "N2TL3" or road_id == "W2TL3":
                    current_phase = ts.get_phase("B1")
                    time_remaining = ts.remaining_phase_time("B1")
                elif road_id == "E2TL4" or road_id == "S2TL4" or road_id == "N2TL4" or road_id == "W2TL4":
                    current_phase = ts.get_phase("A1")
                    time_remaining = ts.remaining_phase_time("A1")

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

                if road_id.startswith("E2TL"):
                    # phase_id_index = 1
                    tmp_obs.append(0) 
                    tmp_obs.append(0) 
                elif road_id.startswith("N2TL"):
                    # phase_id_index = 2
                    tmp_obs.append(0) 
                    tmp_obs.append(1) 
                elif road_id.startswith("W2TL"):
                    # phase_id_index = 1
                    tmp_obs.append(0) 
                    tmp_obs.append(0) 
                elif road_id.startswith("S2TL"):
                    # phase_id_index = 2
                    tmp_obs.append(0) 
                    tmp_obs.append(1) 
                elif road_id.startswith(':TL1') or road_id.startswith(':TL2') or road_id.startswith(':TL3') or road_id.startswith(':TL4'):
                    # middle of intersection 
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
        # this method is written for testing only. Reward function for multi intersection scenario is not defined.
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
    