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


class SimulationStats:
    def __init__(self):
        self.veh_data = {}
        self.veh_fuel_data_avg = []
        self.veh_emission_data_avg = []
        self.veh_speed_data_avg = []

    def reset(self):
        self.veh_data = {}

    def print_stats(self, epi_speeds, epi_fuel):
        print("----------------------------------------")
        print(
            "Avg per vehicle speed (this episode): "
            + str(np.mean(np.array(epi_speeds)))
            + " m/s"
        )
        print(
            "Avg per vehicle trip fuel (this episode): "
            + str(np.mean(np.array(epi_fuel)))
            + " kg"
        )
        print(
            "All vehicle fuel consumption (this episode): "
            + str(np.sum(np.array(epi_fuel)))
            + " kg"
        )
        print("Number of considered vehicles (this episode): " + str(len(epi_fuel)))
        print("")
        print(
            "Avg per vehicle speed (avg over episodes): "
            + str(np.mean(np.array(self.veh_speed_data_avg)))
            + " m/s"
        )
        print(
            "Avg per vehicle fuel (avg over episodes): "
            + str(np.mean(np.array(self.veh_fuel_data_avg)))
            + " kg"
        )
        print(
            "Avg per vehicle co2 emission (avg over episodes): "
            + str(np.mean(np.array(self.veh_emission_data_avg)) / 1000000)
            + " kg/s"
        )
        print("----------------------------------------")
