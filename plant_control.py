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

        self.pid = PID()
        self.pid.sample_time = 1.0 / sample_rate
        self.pid.reset()

        self.sample_rate = sample_rate

    def _step(self, t, r_t, y_t_prev):
        self.pid.setpoint = r_t
        u_t_uncapped = self.pid(y_t_prev)

        u_t = u_t_uncapped
        if u_t < 0.0:
            u_t = 0.0
        if u_t > 100.0:
            u_t = 100.0

        self.plant.U1 = u_t
        y_t  = self.plant.T1
        y2_t = self.plant.T2

        return [t, r_t,
                self.pid.tunings[0], self.pid.tunings[1], self.pid.tunings[2],
                self.pid._proportional, self.pid._integral, self.pid._derivative,
                self.pid._last_error, u_t_uncapped, u_t, y_t,
                0.0, y2_t,
                STATE_NORMAL], y_t

    def episode(self, setpoints, pid_tunings):
        results = pd.DataFrame(columns=EPISODE_COLUMNS)
        
        self.pid.tunings = pid_tunings
        y_t_prev = self.plant.T1
        for t in range(len(setpoints)):
            time.sleep(1.0 / self.sample_rate)
    
            step_data, y_t_prev = self._step(t / self.sample_rate, setpoints[t], y_t_prev)
            results.loc[len(results)] = step_data

        return results


def generate_save_and_plot_episode(plant, setpoint, pid_tunings, directory):
    os.makedirs(directory, exist_ok=True)

    setpoints = np.zeros(EPISODE_LENGTH)
    setpoints[:] = setpoint

    while True:        
        basename = datetime.utcnow().isoformat()
        episode_file = f"{directory}/{basename}Z.parquet"
        episode_plot = f"{directory}/{basename}Z.png"

        print(f"generating episode {basename}...")

        results = plant_control.episode(setpoints, pid_tunings)
        results.to_parquet(episode_file)
        plot_episode(results, episode_plot)
    

plant_control = PlantControl(IS_HARDWARE, SAMPLE_RATE)
generate_save_and_plot_episode(plant_control, 23.0, (50.0, 0.001, 0.1), "episodes")

