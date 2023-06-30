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


import subprocess
import sumolib
import traci
import traci.constants as T  # https://sumo.dlr.de/pydoc/traci.constants.html
from traci.exceptions import FatalTraCIError, TraCIException
import warnings
from sumo_utils.general import *


V = Namespace(
    **{
        k[4:].lower(): k
        for k, v in inspect.getmembers(T, lambda x: not callable(x))
        if k.startswith("VAR_")
    }
)
TL = Namespace(
    **{
        k[3:].lower(): k
        for k, v in inspect.getmembers(T, lambda x: not callable(x))
        if k.startswith("TL_")
    }
)

COLLISION = Namespace(teleport="teleport", warn="warn", none="none", remove="remove")


class SumoDef:
    """
    Start the SUMO simulation as a subprocess
    """

    config_args = dict(net="net-file")
    file_args = set(config_args.keys())

    def __init__(self, c):
        self.c = c
        self.dir = c.get("sumo_dir")
        self.sumo_cmd = None

    def return_sumo_net(self, **kwargs):
        net_path = self.dir + "/net.xml"
        return net_path

    def generate_sumo(self, **kwargs):
        """
        Generate sumo instance using provided configurations
        """
        c = self.c
        # https://sumo.dlr.de/docs/SUMO.html
        sumo_args = Namespace(
            **{arg: kwargs[k] for k, arg in SumoDef.config_args.items() if k in kwargs},
            **kwargs.get("sumo_args", {}),
        ).setdefaults(
            **{
                "begin": 0,
                # 'num-clients': 1,
                "step-length": c.sim_step,
                "no-step-log": True,
                "time-to-teleport": -1,
                "no-warnings": c.get("no_warnings", True),
                "collision.action": COLLISION.remove,
                "collision.check-junctions": True,
                "max-depart-delay": c.get("max_depart_delay", 0.5),
                "random": True,
                "start": c.get("start", True),
                "emission-output": "run-logs.xml",
            }
        )
        config_path = self.dir + "/sumo_config.sumocfg"
        cmd = ["sumo-gui" if c.render else "sumo", "-c", config_path]
        for k, v in sumo_args.items():

            cmd.extend(
                ["--%s" % k, (str(v).lower() if isinstance(v, bool) else str(v))]
                if v is not None
                else []
            )
        c.log(" ".join(cmd))
        return cmd

    def start_sumo(self, tc, tries=3):
        for _ in range(tries):
            try:
                if tc and not "TRACI_NO_LOAD" in os.environ:
                    tc.load(self.sumo_cmd[1:])
                else:
                    if tc:
                        tc.close()
                    else:
                        self.port = sumolib.miscutils.getFreeSocketPort()
                    # Taken from traci.start but add the DEVNULL here
                    p = subprocess.Popen(
                        self.sumo_cmd + ["--remote-port", f"{self.port}"],
                        **dif(
                            self.c.get("sumo_no_errors", True),
                            stderr=subprocess.DEVNULL,
                        ),
                    )
                    tc = traci.connect(self.port, 10, "localhost", p)
                return tc
            except traci.exceptions.FatalTraCIError:  # Sometimes there's an unknown error while starting SUMO
                if tc:
                    tc.close()
                self.c.log("Restarting SUMO...")
                tc = None


class SubscribeDef:
    """
    SUMO subscription manager
    """

    def __init__(self, tc_module, subs):
        self.tc_mod = tc_module
        self.names = [k.split("_", 1)[1].lower() for k in subs]
        self.constants = [getattr(T, k) for k in subs]

    def subscribe(self, *id):
        self.tc_mod.subscribe(*id, self.constants)
        return self

    def get(self, *id):
        res = self.tc_mod.getSubscriptionResults(*id)
        return Namespace(((n, res[v]) for n, v in zip(self.names, self.constants)))

    def getAll(self):
        res = self.tc_mod.getAllSubscriptionResults()
        return Namespace(
            ((y, n), res[y][v])
            for y in res.keys()
            for n, v in zip(self.names, self.constants)
        )


class Vehicle:
    def __init__(
        self,
        id,
        road_id,
        lane_id,
        route_id,
        laneposition,
        speed,
        position,
        fuelconsumption,
        co2emission,
        accel,
    ):
        self.id = id
        self.road_id = road_id
        self.lane_id = lane_id
        self.route_id = route_id
        self.laneposition = laneposition
        self.speed = speed
        self.accel = accel
        self.position = position
        self.fuelconsumption = fuelconsumption
        self.co2emission = co2emission
        self.total_co2_emission = 0
        self.analytical_co2_emit = (
            9449
            + 938.4 * self.speed * self.accel
            - 467.1 * self.speed
            + 28.26 * (self.speed**3)
        )
        self.analytical_total_co2_emission = self.analytical_co2_emit

    def update(
        self,
        road_id,
        lane_id,
        route_id,
        laneposition,
        speed,
        position,
        fuelconsumption,
        co2emission,
        accel,
    ):
        self.road_id = road_id
        self.lane_id = lane_id
        self.route_id = route_id
        self.laneposition = laneposition
        self.speed = speed
        self.accel = accel
        self.position = position
        self.fuelconsumption = fuelconsumption
        self.co2emission = co2emission
        self.total_co2_emission += co2emission
        # emission class : PC_G_EU4
        self.analytical_co2_emit = (
            9449
            + 938.4 * self.speed * self.accel
            - 467.1 * self.speed
            + 28.26 * (self.speed**3)
        )
        self.analytical_total_co2_emission += self.analytical_co2_emit


class TrafficState:
    """
    Keeps relevant parts of SUMO simulation state in Python for easy access
    """

    def __init__(self, c, tc, net, **kwargs):
        """
        Initialize and populate container objects
        """
        self.c = c
        self.tc = tc
        self.lane_lengths = {}
        self.subscribes = Namespace()
        self.vehicles = {}
        self.completed_vehicle_count = 0
        self.driver_variety = {}

    def step(self):
        """
        Take a simulation step and update state
        """
        subscribes = self.subscribes
        # Actual SUMO step
        self.tc.simulationStep()
        sim_res = subscribes.sim.get()
        self.completed_vehicle_count = 0
        # subscribe to newly departed vehicles
        for veh_id in sim_res.departed_vehicles_ids:
            subscribes.veh.subscribe(veh_id)

        # update vehicle states
        veh_info = subscribes.veh.getAll()
        tmp = self.vehicles.copy()
        for veh_id in tmp.keys():
            if veh_id not in sim_res.arrived_vehicles_ids:

                self.vehicles[veh_id].update(
                    veh_info[(veh_id, "road_id")],
                    veh_info[(veh_id, "lane_id")],
                    veh_info[(veh_id, "route_id")],
                    veh_info[(veh_id, "laneposition")],
                    veh_info[(veh_id, "speed")],
                    veh_info[(veh_id, "position")],
                    veh_info[(veh_id, "fuelconsumption")],
                    veh_info[(veh_id, "co2emission")],
                    veh_info[(veh_id, "acceleration")],
                )
            else:
                self.completed_vehicle_count += 1
                del self.vehicles[veh_id]

        for veh_id in sim_res.departed_vehicles_ids:
            self.vehicles[veh_id] = Vehicle(
                veh_id,
                veh_info[(veh_id, "road_id")],
                veh_info[(veh_id, "lane_id")],
                veh_info[(veh_id, "route_id")],
                veh_info[(veh_id, "laneposition")],
                veh_info[(veh_id, "speed")],
                veh_info[(veh_id, "position")],
                veh_info[(veh_id, "fuelconsumption")],
                veh_info[(veh_id, "co2emission")],
                veh_info[(veh_id, "acceleration")],
            )

    def setup(self):
        """
        Add subscriptions for some SUMO state variables
        """
        tc = self.tc
        subscribes = self.subscribes

        subscribes.sim = SubscribeDef(
            tc.simulation,
            [
                V.departed_vehicles_ids,
                V.arrived_vehicles_ids,
                V.colliding_vehicles_ids,
                V.loaded_vehicles_ids,
            ],
        ).subscribe()
        subscribes.tl = SubscribeDef(tc.trafficlight, [TL.red_yellow_green_state])
        subscribes.veh = SubscribeDef(
            tc.vehicle,
            [
                V.road_id,
                V.lane_id,
                V.route_id,
                V.laneposition,
                V.speed,
                V.acceleration,
                V.position,
                V.fuelconsumption,
                V.co2emission,
            ],
        )

    def reset(self, tc):
        self.tc = tc
        self.subscribes.clear()
        self.vehicles.clear()

    """ Wrapper methods for traci calls to interact with simulation """

    def remove(self, veh_id):
        try:
            self.tc.vehicle.remove(veh_id)
            self.tc.vehicle.unsubscribe(veh_id)
        except TraCIException as e:
            warnings.warn(
                "Received nonfatal error while removing vehicle %s:\n%s" % (veh_id, e)
            )

    def add(
        self, veh_id, route, type, lane_index="first", pos="base", speed=0, patience=3
    ):
        try:
            self.tc.vehicle.add(
                veh_id,
                str(route.id),
                typeID=str(type.id),
                departLane=str(lane_index),
                departPos=str(pos),
                departSpeed=str(speed),
            )
        except TraCIException as e:
            if patience == 0:
                raise FatalTraCIError(
                    "Tried for 3 times to add vehicle but still got error: " + str(e)
                )
            warnings.warn(
                "Caught the following exception while adding vehicle %s, removing and readding vehicle:\n%s"
                % (veh_id, e)
            )
            self.remove(veh_id)
            self.add(veh_id, route, type, lane_index, pos, speed, patience=patience - 1)

    def get_max_speed(self, veh):
        self.tc.vehicle.getMaxSpeed(veh)

    def get_program(self, tl):
        return self.tc.trafficlight.getProgram(tl)

    def get_phase(self, tl):
        return self.tc.trafficlight.getPhase(tl)

    def get_vehicle_list(self):
        return self.vehicles.keys()

    def get_speed(self, veh):
        return self.vehicles[veh].speed

    def get_position(self, veh):
        return self.vehicles[veh].laneposition

    def get_edge(self, veh):
        return self.vehicles[veh].road_id
    
    def get_lane_length(self, lane):
        return self.tc.lane.getLength(lane)

    def get_dist_intersection(self, veh, lane_length):
        lane_id = self.vehicles[veh].lane_id
        lane_pos = self.vehicles[veh].laneposition
        # invert lane position value,
        # so if the car is close to the traffic light -> lane_pos = 0 -> 750 = max len of a road
        # https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getLanePosition
        return lane_length - lane_pos, lane_id

    def get_acceleration(self, veh):
        return self.vehicles[veh].accel

    def get_lane_id(self, veh):
        return self.vehicles[veh].lane_id

    def remaining_phase_time(self, tl):
        return self.tc.trafficlight.getNextSwitch(tl) - self.tc.simulation.getTime()

    def get_co2_emission(self, veh):
        return self.vehicles[veh].co2emission

    def get_analytical_co2_emission(self, veh):
        return self.vehicles[veh].analytical_co2_emit

    def get_leader_alternative(self, veh):
        lane_id = self.vehicles[veh].lane_id
        lane_pos = self.vehicles[veh].laneposition

        candidates = {}

        for veh_id in self.vehicles.keys():
            v_lane_id = self.vehicles[veh_id].lane_id
            if v_lane_id == lane_id:
                v_lane_pos = self.vehicles[veh_id].laneposition
                if v_lane_pos >= lane_pos:
                    #print(f"my id {veh} and leader {veh_id}.")
                    #print(f"my pos {lane_pos} and leader pos {v_lane_pos}.")
                    candidates[veh_id] = v_lane_pos - lane_pos
        # if empty, return None
        if not bool(candidates):
            return None
        sorted_candidates = {
            k: v for k, v in sorted(candidates.items(), key=lambda item: item[1])
        }
        keys = list(sorted_candidates)
        if len(keys) == 1:
            return None
        return keys[1], sorted_candidates[keys[1]]

    def get_leader(self, veh):
        lane_id = self.vehicles[veh].lane_id
        lane_pos = self.vehicles[veh].laneposition
        route_id = self.vehicles[veh].route_id
        incoming_lane_lebgth = self.tc.lane.getLength(lane_id)
        intersection_length = self.tc.lane.getLength(":TL_1_0")
        veh_length = self.tc.vehicle.getLength(veh)          
        candidates = {}

        for veh_id in self.vehicles.keys():
            v_lane_id = self.vehicles[veh_id].lane_id
            v_route_id = self.vehicles[veh_id].route_id
            if v_lane_id == lane_id:
                v_lane_pos = self.vehicles[veh_id].laneposition
                if v_lane_pos >= lane_pos:
                    candidates[veh_id] = v_lane_pos - lane_pos - veh_length
            elif v_route_id == route_id:
                v_lane_pos = self.vehicles[veh_id].laneposition
                if v_lane_id == "TL2S_0" or v_lane_id == "TL2N_0" or v_lane_id == "TL2W_0" or v_lane_id == "TL2E_0":
                    candidates[veh_id] = incoming_lane_lebgth + intersection_length + v_lane_pos - lane_pos - veh_length

        # if empty, return None
        if not bool(candidates):
            return None
        sorted_candidates = {
            k: v for k, v in sorted(candidates.items(), key=lambda item: item[1])
        }
        keys = list(sorted_candidates)
        if len(keys) == 1:
            return None
        return keys[1], sorted_candidates[keys[1]]

    def get_follower(self, veh):
        lane_id = self.vehicles[veh].lane_id
        lane_pos = self.vehicles[veh].laneposition
        route_id = self.vehicles[veh].route_id
        incoming_lane_lebgth = self.tc.lane.getLength(lane_id)
        intersection_length = self.tc.lane.getLength(":TL_1_0")
        veh_length = self.tc.vehicle.getLength(veh)    

        candidates = {}

        for veh_id in self.vehicles.keys():
            v_lane_id = self.vehicles[veh_id].lane_id
            v_route_id = self.vehicles[veh_id].route_id
            if v_lane_id == lane_id:
                v_lane_pos = self.vehicles[veh_id].laneposition
                if v_lane_pos <= lane_pos:
                    candidates[veh_id] = lane_pos - v_lane_pos - veh_length
            elif v_route_id == route_id:
                v_lane_pos = self.vehicles[veh_id].laneposition
                if v_lane_id == "S2TL_0" or v_lane_id == "E2TL_0" or v_lane_id == "W2TL_0" or v_lane_id == "N2TL_0":
                    candidates[veh_id] = incoming_lane_lebgth + intersection_length + lane_pos - v_lane_pos - veh_length
        # if empty, return None
        if not bool(candidates):
            return None
        sorted_candidates = {
            k: v for k, v in sorted(candidates.items(), key=lambda item: item[1])
        }
        keys = list(sorted_candidates)
        if len(keys) == 1:
            return None
        return keys[1], sorted_candidates[keys[1]]

    def get_follower_alternative(self, veh):
        lane_id = self.vehicles[veh].lane_id
        lane_pos = self.vehicles[veh].laneposition

        candidates = {}

        for veh_id in self.vehicles.keys():
            v_lane_id = self.vehicles[veh_id].lane_id
            if v_lane_id == lane_id:
                v_lane_pos = self.vehicles[veh_id].laneposition
                if v_lane_pos <= lane_pos:
                    candidates[veh_id] = lane_pos - v_lane_pos
        # if empty, return None
        if not bool(candidates):
            return None
        sorted_candidates = {
            k: v for k, v in sorted(candidates.items(), key=lambda item: item[1])
        }
        keys = list(sorted_candidates)
        if len(keys) == 1:
            return None
        return keys[1], sorted_candidates[keys[1]]

    def set_speed(self, veh, speed):
        self.tc.vehicle.setSpeed(veh, speed)

    def set_accel(self, veh, accel):
        self.tc.vehicle.setAccel(veh, accel)

    def set_phase(self, tl, phase_index):
        return self.tc.trafficlight.setPhase(tl, phase_index)

    def set_program(self, tl, program):
        self.tc.trafficlight.setProgram(tl, program)

    def set_color(self, veh, color):
        self.tc.vehicle.setColor(veh, color + (255,))

    def accel(self, veh, acc, n_acc_steps=1):
        """
        Let the initial speed be v0, the sim_step be dt, and the acceleration be a.
        This function increases v0 over n=n_acc_steps steps by a*dt/n per step.
        At each of the sim steps, the speed increases by a*dt/n at the BEGINNING of the step.
        After one step, the vehicle's speed is v1=v0+a*dt/n and the distance traveled is v1*dt.
        If n>1, then after two steps, the vehicle's speed is v2=v1+a*dt/n and the distance traveled is v2*dt. Etc etc.
        If accel is called again before n steps has elapsed, the new acceleration action
        overrides the continuation of any previous acceleration.
        The per step acceleration a/n is clipped by SUMO's IDM behavior to be in the range
        of -max_decel <= a/n <= max_accel, where max_accel and max_decel are the IDM parameters given to SUMO.
        """
        speed = self.get_speed(veh)
        self.tc.vehicle.slowDown(veh, max(0, speed + acc * self.c.sim_step), 1e-3)

    def set_max_speed(self, veh, speed):
        self.tc.vehicle.setMaxSpeed(veh, max(speed, 1e-3))

    def slow_down(self, veh, speed, duration):
        self.tc.vehicle.slowDown(veh, speed, duration)

    def lane_change(self, veh, lane_index, direction):
        assert direction in [-1, 0, 1]
        self.tc.vehicle.changeLane(veh, lane_index + int(direction), 100000.0)

    def lane_change_to(self, veh, lane_index):
        self.tc.vehicle.changeLane(veh, lane_index, 100000.0)

    def get_co2_emission(self, veh):
        return self.tc.vehicle.getCO2Emission(veh)

    def set_IDM_accel(self, veh, noise=False, variety=False):
        v0 = 30
        T = 1
        a = 1
        b = 1.5
        delta = 4
        s0 = 1.5

        if variety:
            if veh not in self.driver_variety.keys():
                v0 = np.random.normal(v0, 5)
                T = np.random.normal(T, 0.2)
                a = np.random.normal(a, 0.2)
                b = np.random.normal(b, 0.2)
                delta = delta
                s0 = np.random.normal(s0, 0.2)
                self.driver_variety[veh] = [v0, T, a, b, delta, s0]
            else:
                v0 = self.driver_variety[veh][0]
                T = self.driver_variety[veh][1]
                a = self.driver_variety[veh][2]
                b = self.driver_variety[veh][3]
                delta = self.driver_variety[veh][4]
                s0 = self.driver_variety[veh][5]

        v = self.get_speed(veh)
        leader_info = self.get_leader(veh)

        if leader_info is None or leader_info == '':  # no car ahead
            s_star = 0
            h = 1e-10
        else:
            lead_id, h = leader_info
            lead_vel = self.get_speed(lead_id)
            s_star = s0 + max(0, v * T + (v * (v - lead_vel) /(2 * np.sqrt(a * b))))
            
        accel = a * (1 - (v / v0)**delta - (s_star/h)**2)
        if noise:
            n = np.random.uniform(-0.2, 0.2)
            accel += n
        self.tc.vehicle.slowDown(veh, max(0, v + accel * self.c.sim_step), 0)
        