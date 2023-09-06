#!/usr/bin/env python
#
# A script that runs the control loop under supervision. Where the plant control
# worked in steps, the supervisor works in episodes. It is responsible to check
# that each episode the system remains stable, and it can revert the control
# loop to known-stable (though suboptimal) PID parameters.
#

import numpy as np
import pandas as pd
from datetime import datetime

from episodes import SAMPLE_RATE, EPISODE_LENGTH, COL_ERROR, EPISODE_COLUMNS, STATE_NORMAL, STATE_FALLBACK, save_and_plot_episode
from plant_control import PlantControl


IS_HARDWARE = False

PID_TUNINGS = (50.0, 0.001, 0.1)
SET_POINT = 23.0

BENCHMARK_ERROR = 9.0
FALLBACK_PID_TUNINGS = (20.0, 0.1, 0.01)

class SupervisedPlantControl:
    def __init__(self, plant, R_bmk, fallback_pid_tunings):
        self.plant = plant
        self.sample_rate = plant.sample_rate

        self.R_bmk = R_bmk
        self.fallback_pid_tunings = fallback_pid_tunings


    def episode(self, setpoints, pid_tunings):
        results = pd.DataFrame(columns=EPISODE_COLUMNS)

        episode_state = STATE_NORMAL
        self.plant.set_pid_tunings(pid_tunings, "episode starts")

        for t in range(len(setpoints)):
            step_data = self.plant.step(t / self.sample_rate, setpoints[t],
                                        R_bmk=self.R_bmk, episode_state=episode_state)
            results.loc[len(results)] = step_data

            # in fallback state we just sit the episode out
            running_error = (results[COL_ERROR]**2).sum()
            if episode_state != STATE_FALLBACK and \
                    running_error > self.R_bmk:
                episode_state = STATE_FALLBACK
                self.plant.set_pid_tunings(self.fallback_pid_tunings,
                                           f"running error {running_error:.1f} exceeds benchmark error {self.R_bmk:.1f}")

        return results


#
# The main driver, create a supervised plant-control triplet, set the set-points
# and PID tunings and run episodes until the program is stopped.
#
if __name__ == "__main__":
    setpoints = np.zeros(EPISODE_LENGTH)
    setpoints[:] = SET_POINT

    plant_control = PlantControl(IS_HARDWARE, SAMPLE_RATE)
    plant_control.set_pid_tunings(FALLBACK_PID_TUNINGS, "program starts")
    supervised_plant_control = SupervisedPlantControl(plant_control, BENCHMARK_ERROR, FALLBACK_PID_TUNINGS)
    while True:
        timestamp_utc = datetime.utcnow()
        print(f"generating episode {timestamp_utc.isoformat()}...")

        episode = supervised_plant_control.episode(setpoints, PID_TUNINGS)
        save_and_plot_episode(timestamp_utc, episode)

