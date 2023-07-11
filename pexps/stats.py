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


from statistics import mean
from imported_utils import *
from exp import *
from env import *


class SimulationStats:
    def __init__(self):
        self.veh_data = {}
        self.running_stats = {}

        # per vehicle properties
        self.veh_fuel_data_avg = []
        self.veh_emission_data_avg = []
        self.veh_speed_data_avg = []
        # per fleet properties
        self.veh_fuel_sum_data = []
        self.veh_emission_sum_data = []

    def record_running_stats(self, stats_summary):
        for key, value in stats_summary[0].items():
            if key not in self.running_stats:
                self.running_stats[key] = [value]
            else:
                self.running_stats[key].append(value)

    def store_stats(self, stats_summary={}, wandb_record=False, record_step=0):
        # update running stats
        self.record_running_stats(stats_summary)
        # for every key in stats_summary, store the value in the corresponding list
        print("--------------------")
        print("Current episode stats")
        for key, value in stats_summary[0].items():
            if key == "speed":
                print("Average speed", value)
            elif key == "fuel":
                print("Average fuel", value)
            elif key == "emission":
                print("Average emission", value)
            elif key == "fuel_sum":
                print("Total fuel", value)
            elif key == "emission_sum":
                print("Total emission", value)
            elif key == "steps":
                continue
            else:
                print(key, value)
        print("--------------------")
        print("Running stats")

        for key, value in self.running_stats.items():
            if key == "speed":
                print("Average speed", mean(value))
            elif key == "fuel":
                print("Average fuel", mean(value))
            elif key == "emission":
                print("Average emission", mean(value))
            elif key == "fuel_sum":
                print("Total fuel", mean(value))
            elif key == "emission_sum":
                print("Total emission", mean(value))
            elif key == "steps":
                continue
            else:
                print(key, mean(value))
        print("--------------------")
        # record the stats in wandb
        self.wandb_record(stats_summary, wandb_record, record_step)

    def wandb_record(self, stats_summary, wandb_record, record_step):
        try:
            if wandb_record:
                # log the training progress in wandb
                wandb.log(
                    {
                        "Average per vehicle speed": stats_summary[0]["speed"],
                        "Average per vehicle fuel": stats_summary[0]["fuel"],
                        "Average per vehicle emission": stats_summary[0]["emission"],
                        "Total fleet fuel": stats_summary[0]["fuel_sum"],
                        "Total fleet emission": stats_summary[0]["emission_sum"],
                    },
                    step=record_step,
                )

        except KeyError:
            print("Wandb recording error!!")
