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


    def episode(self, setpoints, pid_tunings):
        results = pd.DataFrame(columns=EPISODE_COLUMNS)

        episode_state = STATE_NORMAL
        if pid_tunings is not None:
            print(f"at 0: setting new episode PID parameters {pid_tunings}")
            self.plant.set_pid_tunings(pid_tunings)

        for t in range(len(setpoints)):
            time.sleep(1.0 / self.sample_rate)

            step_data = self.plant.step(t / self.sample_rate, setpoints[t], episode_state)
            results.loc[len(results)] = step_data

            # in fallback state we just sit the episode out
            if episode_state != STATE_FALLBACK and \
                    (results[COL_ERROR]**2).sum() > self.lambda_benchmark_error:
                print(f"at {t}: running error {(results[COL_ERROR]**2).sum():.1f} above lamda*benchmark error {self.lambda_benchmark_error:.1f}, switched to fall-back PID parameters {self.fallback_pid_tunings}")
                episode_state = STATE_FALLBACK
                self.plant.set_pid_tunings(self.fallback_pid_tunings)

        if episode_state == STATE_FALLBACK:
            print(f"at {t}: episode ended, restoring normal PID parameters {pid_tunings}")
            self.plant.set_pid_tunings(pid_tunings)

        return results


def generate_save_and_plot_episode(plant, setpoint, pid_tunings, directory):
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
        pid_tunings = None


if __name__ == "__main__":
    plant_control = PlantControl(IS_HARDWARE, SAMPLE_RATE)
    supervised_plant_control = SupervisedPlantControl(plant_control, 1200.0, 1.0, (20.0, 0.1, 0.01))
    generate_save_and_plot_episode(supervised_plant_control, 23.0, (50.0, 0.001, 0.1), "episodes")

