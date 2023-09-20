#!/usr/bin/env python
#
# A script that runs the control loop under supervision. Where the plant control
# worked in steps, the supervisor works in episodes. It is responsible to check
# that each episode the system remains stable, and it can revert the control
# loop to known-stable (though suboptimal) PID parameters.
#

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

        self.t = EPISODE_LENGTH # so we start a new episode on the next step

        self.episode_state = STATE_NORMAL
        self.R_bmk = R_bmk

        self.fallback_pid_tunings = fallback_pid_tunings
        self.proposed_pid_tunings = fallback_pid_tunings


    def start_episode(self):
        self.t = 0
        self.results = pd.DataFrame(columns=EPISODE_COLUMNS)
        self.episode_state = STATE_NORMAL
        self.plant.set_pid_tunings(self.proposed_pid_tunings, "episode starts")


    # note that these tunings will only be applied at the start of an episode;
    # i.e. when $t$ is 0.
    def set_pid_tunings(self, pid_tunings):
        self.proposed_pid_tunings = pid_tunings


    def step(self, setpoint):
        if self.t == EPISODE_LENGTH:
            self.start_episode()
        else:
            self.t += 1

        step_data = self.plant.step(self.t / SAMPLE_RATE, setpoint,
                                    R_bmk=self.R_bmk, episode_state=self.episode_state)
        self.results.loc[len(self.results)] = step_data

        # in fallback state we just sit the episode out
        running_error = (self.results[COL_ERROR]**2).sum()
        if self.episode_state != STATE_FALLBACK and running_error > self.R_bmk:
            self.episode_state = STATE_FALLBACK
            self.plant.set_pid_tunings(self.fallback_pid_tunings,
                                       f"running error {running_error:.1f} exceeds benchmark error {self.R_bmk:.1f}")

        return self.results, self.t == EPISODE_LENGTH


#
# The main driver, create a supervised plant-control triplet, set the set-points
# and PID tunings and run episodes until the program is stopped.
#
if __name__ == "__main__":
    plant_control = PlantControl(IS_HARDWARE, FALLBACK_PID_TUNINGS)

    supervised_plant_control = SupervisedPlantControl(plant_control, BENCHMARK_ERROR, FALLBACK_PID_TUNINGS)
    supervised_plant_control.set_pid_tunings(PID_TUNINGS)

    while True:
        episode, done = supervised_plant_control.step(SET_POINT)
        if done:
            timestamp_utc = datetime.utcnow() # XXX push into episode
            print(f"saving episode {timestamp_utc.isoformat()}...")
            save_and_plot_episode(timestamp_utc, episode)

