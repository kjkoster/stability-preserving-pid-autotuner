#!/usr/bin/env python
#
# A script that runs the control loop under supervision.
#

import os
import time
import numpy as np
import pandas as pd
from simple_pid import PID
from datetime import datetime


from safe_pid_autotuner.safe_pid_tuner import SAMPLE_RATE, EPISODE_LENGTH, COL_ERROR, EPISODE_COLUMNS, STATE_NORMAL, STATE_FALLBACK, plot_episode
from plant_control import PlantControl

IS_HARDWARE = False

class SupervisedPlantControl:
    def __init__(self, plant, benchmark_error, lamba_error, fallback_pid_tunings):
        self.plant = plant
        self.sample_rate = plant.sample_rate

        self.lambda_benchmark_error = lamba_error * benchmark_error
        self.fallback_pid_tunings = fallback_pid_tunings


    # this overrides the PlantControl::episode()
    def episode(self, setpoints, pid_tunings):
        results = pd.DataFrame(columns=EPISODE_COLUMNS)

        episode_state = STATE_NORMAL
        self.plant.set_pid_tunings(pid_tunings, "episode starts")

        for t in range(len(setpoints)):
            self.plant.sleep_until_cycle_starts()

            step_data = self.plant.step(t / self.sample_rate, setpoints[t], episode_state)
            results.loc[len(results)] = step_data

            # in fallback state we just sit the episode out
            running_error = (results[COL_ERROR]**2).sum()
            if episode_state != STATE_FALLBACK and \
                    running_error > self.lambda_benchmark_error:
                episode_state = STATE_FALLBACK
                self.plant.set_pid_tunings(self.fallback_pid_tunings,
                                           f"running error {running_error:.1f} exceeds benchmark error {self.lambda_benchmark_error:.1f}")

        return results


def generate_save_and_plot_episodes(plant, setpoint, pid_tunings, directory):
    os.makedirs(directory, exist_ok=True)

    setpoints = np.zeros(EPISODE_LENGTH)
    setpoints[:] = setpoint

    while True:
        basename = datetime.utcnow().isoformat()
        episode_file = f"{directory}/{basename}Z.parquet"
        episode_plot = f"{directory}/{basename}Z.png"

        print(f"supervised: generating episode {basename}...")

        results = plant.episode(setpoints, pid_tunings)
        results.to_parquet(episode_file)
        plot_episode(results, episode_plot)

        # this is where we'd ask the autotuner, but that is for later
        # pid_tunings = ask_autotuner()


if __name__ == "__main__":
    plant_control = PlantControl(IS_HARDWARE, SAMPLE_RATE)
    supervised_plant_control = SupervisedPlantControl(plant_control, 9.0, 1.0, (20.0, 0.1, 0.01))
    generate_save_and_plot_episodes(supervised_plant_control, 23.0, (50.0, 0.001, 0.1), "episodes")

