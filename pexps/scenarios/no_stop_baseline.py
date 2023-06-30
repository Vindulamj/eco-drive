from concurrent.futures import process
from numpy.lib.polynomial import roots
from imported_utils import *
from exp import *
from env import *
import sympy

# Vehicle colors
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
YELLOW = (255, 255, 51)
RED = (255, 0, 0)

TOL = 0  # tolerance around freewheeling time
CHANGE_SPEED_DUR = 0.1

YELLOW_DURATION = 4
GREEN_DURATION = 30


class NoStopBaselineEnv(Env):
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
                act, enable = self.eco_cacc(rl_vehs[i], t)

        super().step()
        # call this after step() function to get updated vehicle list
        vehicle_list = ts.get_vehicle_list()
        obs, ids = self.get_state(vehicle_list, c, ts)
        reward, _ = self.get_reward(vehicle_list, c, ts, rl_vehs, veh_edge)
        self.collect_stats(warmup, c, ts, vehicle_list, reward)

        return Namespace(obs=obs, id=ids, reward=reward)

    def eco_cacc(self, veh, t):
        c = self.c
        ts = self.ts
        # set actions for rl vehicles
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]

        # "TL" is common to all vehicles
        current_phase = ts.get_phase("TL")
        steps_todo = ts.remaining_phase_time("TL")

        car_id = veh

        ts.set_color(car_id, RED)
        road_id = ts.get_edge(car_id)
        max_speed = c.target_vel
        cur_speed = ts.get_speed(car_id)
        lane_pos = ts.get_position(car_id)
        # invert lane position value,
        # so if the car is close to the traffic light -> lane_pos = 0 -> 750 = max len of a road
        # https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getLanePosition
        lane_pos = 250 - lane_pos

        # identify which phase id the current vehicle needs to be given green so that it can travel
        phase_match = 0
        if road_id == "S2TL":
            phase_match = 0
        elif road_id == "W2TL":
            phase_match = 2
        elif road_id == "N2TL":
            phase_match = 0
        elif road_id == "E2TL":
            phase_match = 2

        if road_id not in incoming_roads:
            ts.slow_down(car_id, max_speed, CHANGE_SPEED_DUR)
            return -1, False

        time_to_intersection = lane_pos / (cur_speed + 1e-6)

        # Should the vehicle increase its speed?
        # Current phase
        if current_phase == phase_match:
            # check if the current speed is good enough
            step_to_phase_end = steps_todo
            if step_to_phase_end > time_to_intersection + TOL:
                return -1, False
            # check if max speed would make it through
            elif step_to_phase_end > lane_pos / max_speed + TOL:
                ts.slow_down(car_id, max_speed, CHANGE_SPEED_DUR)
                ts.set_color(car_id, CYAN)
                return -1, False

            step_to_phase_start_extended = steps_todo + YELLOW_DURATION * 2 + \
                                                           GREEN_DURATION 
            step_to_phase_end_extended = steps_todo + YELLOW_DURATION * 2 + \
                                                           GREEN_DURATION * 2
            if (step_to_phase_end_extended > time_to_intersection + TOL 
                            and time_to_intersection > step_to_phase_start_extended):
                ts.set_color(car_id, YELLOW)
                return -1, False
            # check if max speed is good enough and if so, find
            # the right speed to reach the intersection at the phase start
            elif step_to_phase_end_extended > lane_pos / max_speed + TOL:
                new_target_speed = lane_pos / step_to_phase_start_extended
                ts.slow_down(car_id, new_target_speed, CHANGE_SPEED_DUR)
                ts.set_color(car_id, CYAN)
                return -1, False

        # Penultimate phase
        if (current_phase + 1) % 4 == phase_match:
            # check if the current speed is good enough
            step_to_phase_start = steps_todo
            step_to_phase_end = GREEN_DURATION + step_to_phase_start

            if (step_to_phase_end > time_to_intersection + TOL 
                    and time_to_intersection > step_to_phase_start):
                ts.set_color(car_id, YELLOW)
                return -1, False
            # check if max speed is good enough and if so, find
            # the right speed to reach the intersection at the phase start
            elif step_to_phase_end > lane_pos / max_speed + TOL:
                if step_to_phase_start != 0:
                    new_target_speed = lane_pos / step_to_phase_start
                    ts.slow_down(car_id, new_target_speed, CHANGE_SPEED_DUR)
                    ts.set_color(car_id, CYAN)
                    return -1, False
                else:
                    return -1, False

        # Pre-Penultimate phase
        if (current_phase + 2) % 4 == phase_match:
            # check if the current speed is good enough
            step_to_phase_start = YELLOW_DURATION + steps_todo
            step_to_phase_end = GREEN_DURATION + step_to_phase_start 

            if (step_to_phase_end > time_to_intersection + TOL
                        and time_to_intersection > step_to_phase_start):
                ts.set_color(car_id, YELLOW)
                return -1, False
            # check if max speed is good enough and if so, find
            # the right speed to reach the intersection at the phase start
            elif step_to_phase_end > lane_pos / max_speed + TOL:
                if step_to_phase_start != 0:
                    new_target_speed = lane_pos / step_to_phase_start
                    ts.slow_down(car_id, new_target_speed, CHANGE_SPEED_DUR)
                    ts.set_color(car_id, CYAN)
                    return -1, False
                else:
                    return -1, False

        # Pre-Pre-Penultimate phase
        if (current_phase + 3) % 4 == phase_match:
            # check if the current speed is good enough
            step_to_phase_start = steps_todo + GREEN_DURATION + YELLOW_DURATION
            step_to_phase_end = GREEN_DURATION + step_to_phase_start

            if ( step_to_phase_end > time_to_intersection + TOL 
                    and time_to_intersection > step_to_phase_start):
                ts.set_color(car_id, YELLOW)
                return -1, False
            # check if max speed is good enough and if so, find
            # the right speed to reach the intersection at the phase start
            elif step_to_phase_end > lane_pos / max_speed + TOL:
                if step_to_phase_start != 0:
                    new_target_speed = lane_pos / step_to_phase_start
                    ts.slow_down(car_id, new_target_speed, CHANGE_SPEED_DUR)
                    ts.set_color(car_id, CYAN)
                    return -1, False
                else:
                    return -1, False
        # If we reach this line, then this vehicle cannot make it to the intersection
        # Use SUMO's default driving behavior
        ts.set_color(car_id, YELLOW)
        return -1, False


    def get_state(self, vehicle_list, c, ts):
        """Placeholder state information"""
        obs = {}

        for veh in vehicle_list:

            if veh.startswith("rl"):
                obs[veh] = np.zeros(c._n_obs)

        sort_id = lambda d: [v for k, v in sorted(d.items())]
        ids = sorted(obs)
        obs = arrayf(sort_id(obs)).reshape(-1, c._n_obs)
        return obs, ids

    def fuel_model(self, v_speed, v_accel):
        """VT-CPFM Fuel Model"""
        R_f = (
            1.23 * 0.6 * 0.98 * 3.28 * (v_speed**2)
            + 9.8066 * 3152 * (1.75 / 1000) * 0.033 * v_speed
            + 9.8066 * 3152 * (1.75 / 1000) * 0.033
            + 9.8066 * 3152 * 0
        )
        power = ((R_f + 1.04 * 3152 * v_accel) / (3600 * 0.92)) * v_speed
        fuel = 0
        if power >= 0:
            fuel = 0.00078 + 0.000019556 * power + 0.000001 * (power**2)
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
