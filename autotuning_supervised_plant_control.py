#!/usr/bin/env python
#
# This is the auto-tuner that generates new PID parameters. The auto-tuner is
# kept separate from the plant control. It works solely on the data that comes
# out of the running, supervised plant control and proposed alternative PID
# tunings for it.
#

import random
import numpy as np
from datetime import datetime

from episodes import SAMPLE_RATE, EPISODE_LENGTH, save_and_plot_episode
from plant_control import PlantControl
from supervised_plant_control import SupervisedPlantControl

IS_HARDWARE = False

PID_TUNINGS = (50.0, 0.001, 0.1)
SET_POINT = 23.0

BENCHMARK_ERROR = 15.0
FALLBACK_PID_TUNINGS = (20.0, 0.1, 0.01)

class PidAutotuner:
    def evaluate_episode(self, episode):
        # to test the workings of the system, we just use random PID settings.
        # This allows us to see everything working without having to dive into
        # machine learning just yet.
        return (random.uniform(0.0, 100.0), random.uniform(0.01, 0.99), random.uniform(0.001, 0.999))

#
# The main driver. Create a supervised plan control and an autotuner. Then start
# running episodes and evaluating these with the autotuner.
#
if __name__ == "__main__":
    setpoints = np.zeros(EPISODE_LENGTH)
    setpoints[:] = SET_POINT

    plant_control = PlantControl(IS_HARDWARE, SAMPLE_RATE)
    plant_control.set_pid_tunings(FALLBACK_PID_TUNINGS, "program starts")
    supervised_plant_control = SupervisedPlantControl(plant_control, BENCHMARK_ERROR, FALLBACK_PID_TUNINGS)
    pid_autotuner = PidAutotuner()

    # knowing nothing, we just start with the fallback tunings
    pid_tunings = FALLBACK_PID_TUNINGS

    while True:
        timestamp_utc = datetime.utcnow()
        print(f"generating episode {timestamp_utc.isoformat()}...")

        episode = supervised_plant_control.episode(setpoints, pid_tunings)
        save_and_plot_episode(timestamp_utc, episode)

        pid_tunings = pid_autotuner.evaluate_episode(episode)

