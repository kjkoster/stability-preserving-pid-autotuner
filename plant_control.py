#!/usr/bin/env python
#
# A script that runs a control loop asynchronously.
#
# The whole thing is wrapped into a class. In earlier incarnations of this code,
# we found that the plant and the controller's internal states need to be kept
# consistent. This also makes it easy to initialise a clean controller-plant
# pair for quick experiments. The code runs iteratively, so that the interfaces
# to the components are developed to be usable in a live environment. Much as I
# like matrix processing and its efficiency, the model does not fit the
# continuous control loop that is common for live systems.
#

import os
import time
import tclab
import numpy as np
import pandas as pd
from simple_pid import PID
from datetime import datetime

from safe_pid_autotuner.safe_pid_tuner import SAMPLE_RATE, EPISODE_LENGTH, EPISODE_COLUMNS, STATE_NORMAL, plot_episode

IS_HARDWARE = False

class PlantControl:
    def __init__(self, is_hardware, sample_rate):
        TCLab = tclab.setup(connected=is_hardware)
        self.plant = TCLab()
        self.y_t_prev = self.plant.T1

        self.pid = PID()

        self.sample_rate = sample_rate
        self.cycle_time = 1.0 / self.sample_rate
        self.pid.sample_time = self.cycle_time
        print(f"sample rate is {self.sample_rate} Hz, cycle time is {self.pid.sample_time} second")

        self.previous_time = None


    def set_pid_tunings(self, pid_tunings, reason):
        if pid_tunings == self.pid.tunings:
            print(f"though {reason}, skip setting PID parameters {pid_tunings} as these already apply")
        else:
            print(f"setting PID parameters (Kp, Ki, Kd) to {pid_tunings}, because {reason}")
            self.pid.tunings = pid_tunings
            self.pid.reset()


    def step(self, t, r_t, episode_state=STATE_NORMAL):
        self.pid.setpoint = r_t
        u_t_uncapped = self.pid(self.y_t_prev)

        u_t = u_t_uncapped
        if u_t < 0.0:
            u_t = 0.0
        if u_t > 100.0:
            u_t = 100.0

        self.plant.U1 = u_t
        y_t  = self.plant.T1
        y2_t = self.plant.T2

        self.y_t_prev = y_t
        return [t, r_t,
                self.pid.Kp, self.pid.Ki, self.pid.Kd,
                self.pid._proportional, self.pid._integral, self.pid._derivative,
                self.pid._last_error, u_t_uncapped, u_t, y_t,
                0.0, y2_t,
                episode_state]


    def sleep_until_cycle_starts(self):
        if self.previous_time is None:
            self.previous_time = time.time()

        current_time = time.time()
        while current_time - self.previous_time < self.cycle_time:
            time.sleep(0.001)
            current_time = time.time()

        if current_time - self.previous_time > self.cycle_time * 1.01:
            print(f"cycle time {current_time - self.previous_time:0.3f} exceeds expected time {self.cycle_time:0.3f}")
        self.previous_time = current_time


    def episode(self, setpoints):
        results = pd.DataFrame(columns=EPISODE_COLUMNS)
        for t in range(len(setpoints)):
            self.sleep_until_cycle_starts()

            step_data = self.step(t / self.sample_rate, setpoints[t])
            results.loc[len(results)] = step_data

        return results


def generate_save_and_plot_episodes(plant, setpoint, pid_tunings, directory):
    os.makedirs(directory, exist_ok=True)

    setpoints = np.zeros(EPISODE_LENGTH)
    setpoints[:] = setpoint

    plant.set_pid_tunings(pid_tunings, "episode starts")

    while True:
        basename = datetime.utcnow().isoformat()
        episode_file = f"{directory}/{basename}Z.parquet"
        episode_plot = f"{directory}/{basename}Z.png"

        print(f"generating episode {basename}...")

        results = plant_control.episode(setpoints)
        results.to_parquet(episode_file)
        plot_episode(results, episode_plot)


if __name__ == "__main__":
    plant_control = PlantControl(IS_HARDWARE, SAMPLE_RATE)
    generate_save_and_plot_episodes(plant_control, 23.0, (50.0, 0.001, 0.1), "episodes")

