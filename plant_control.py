#!/usr/bin/env python
#
# A script that runs a PID control loop. This is the inner-most control system,
# directly controlling the plant with the given PID parameters. The plan control
# runs in steps and does not really have the notion of episodes.
#
# The code runs cyclicly, so that its interface is usable in a live environment.
# Much as I like matrix processing and its efficiency, the model does not fit
# the continuous control loop that is common for live systems.
#

import time
import tclab
import numpy as np
import pandas as pd
from simple_pid import PID
from datetime import datetime

from episodes import SAMPLE_RATE, EPISODE_LENGTH, EPISODE_COLUMNS, STATE_NORMAL, save_and_plot_episode


IS_HARDWARE = False

PID_TUNINGS = (50.0, 0.001, 0.1)
SET_POINT = 23.0


class PlantControl:
    def __init__(self, is_hardware):
        TCLab = tclab.setup(connected=is_hardware)
        self.plant = TCLab()
        self.y_t_prev = self.plant.T1

        self.pid = PID()

        self.cycle_time = 1.0 / SAMPLE_RATE
        self.pid.sample_time = self.cycle_time
        print(f"sample rate is {SAMPLE_RATE} Hz, cycle time is {self.pid.sample_time} second")

        self.previous_time = None


    def set_pid_tunings(self, pid_tunings, reason):
        if pid_tunings == self.pid.tunings:
            print(f"though {reason}, skip setting PID parameters {pid_tunings} as these already apply")
        else:
            print(f"setting PID parameters (Kp, Ki, Kd) to {pid_tunings}, because {reason}")
            self.pid.tunings = pid_tunings
            self.pid.reset()


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


    # note that `step()` blocks until the next cycle, to ensure the timings are good
    def step(self, t, r_t, R_bmk=0.0, u2_t=0.0, episode_state=STATE_NORMAL):
        self.sleep_until_cycle_starts()

        self.pid.setpoint = r_t
        u_t_uncapped = self.pid(self.y_t_prev)

        u_t = u_t_uncapped
        if u_t < 0.0:
            u_t = 0.0
        if u_t > 100.0:
            u_t = 100.0

        self.plant.U1 = u_t
        self.plant.U2 = u2_t
        y_t  = self.plant.T1
        y2_t = self.plant.T2

        self.y_t_prev = y_t
        return [t, r_t,
                self.pid.Kp, self.pid.Ki, self.pid.Kd,
                self.pid._proportional, self.pid._integral, self.pid._derivative,
                self.pid._last_error, R_bmk,
                u_t_uncapped, u_t, y_t, u2_t, y2_t,
                episode_state]

#
# The remainder of this file is code to try out the plant control. We will reuse
# the class above for the other control loop experiments. The code below is just
# a quick driver to test the control loop in isolation.
#

#
# Run a single episode of time T.
#
def run_episode(plant_control, setpoints):
    results = pd.DataFrame(columns=EPISODE_COLUMNS)
    for t in range(len(setpoints)):
        step_data = plant_control.step(t / SAMPLE_RATE, setpoints[t])
        results.loc[len(results)] = step_data

    return results

#
# The main driver, create a plant-control pair, set the set-points and PID
# tunings and run episodes until the program is stopped.
#
if __name__ == "__main__":
    setpoints = np.zeros(EPISODE_LENGTH)
    setpoints[:] = SET_POINT

    plant_control = PlantControl(IS_HARDWARE)
    plant_control.set_pid_tunings(PID_TUNINGS, "program starts")

    while True:
        timestamp_utc = datetime.utcnow()
        print(f"generating episode {timestamp_utc.isoformat()}...")

        episode = run_episode(plant_control, setpoints)
        save_and_plot_episode(timestamp_utc, episode)

