#!/usr/bin/env python
#
# This is a random agent, useful for exploration of the action space, under supervision, of course.
#
import numpy as np
from datetime import datetime

from episodes import T, SAMPLE_RATE, EPISODE_LENGTH, COL_CONTROL_VARIABLE, COL_PROCESS_VARIABLE, COL_ERROR, save_and_plot_episode
from plant_control import PlantControl
from supervised_plant_control import SupervisedPlantControl

IS_HARDWARE = False

PID_TUNINGS = (50.0, 0.001, 0.1)
SET_POINT = 23.0

BENCHMARK_ERROR = 15.0
FALLBACK_PID_TUNINGS = (20.0, 0.1, 0.01)

MAP_GAINS = [500.0, 50.0, 5.0]

N_ACTIONS = 3

def map_action_to_pid_tunings(action):
    return (action[0] * MAP_GAINS[0], action[1] * MAP_GAINS[1], action[2] * MAP_GAINS[2])

def evaluate(episode):
    return -(episode[COL_ERROR]**2).sum()

class RandomAgent(object):
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def choose_action(self):
        return np.random.rand(self.n_actions)

#
# The main driver. Create a supervised plan control and an agent. Prime the
# learning process, then start running episodes and evaluating these with the
# auto-tuner.
#
if __name__ == "__main__":
    setpoints = np.zeros(EPISODE_LENGTH)
    setpoints[:] = SET_POINT

    plant_control = PlantControl(IS_HARDWARE, SAMPLE_RATE)
    plant_control.set_pid_tunings(FALLBACK_PID_TUNINGS, "program starts")
    supervised_plant_control = SupervisedPlantControl(plant_control, BENCHMARK_ERROR, FALLBACK_PID_TUNINGS)
    agent = RandomAgent(N_ACTIONS)

    # knowing nothing, we just start with the fallback tunings
    pid_tunings = FALLBACK_PID_TUNINGS

    # run a first episode, we need a first episode to prime the learning cycle
    timestamp_utc = datetime.utcnow()
    print(f"generating priming episode {timestamp_utc.isoformat()}...")
    episode = supervised_plant_control.episode(setpoints, pid_tunings)

    while True:
        timestamp_utc = datetime.utcnow()
        print(f"generating episode {timestamp_utc.isoformat()}...")

        action = agent.choose_action()
        pid_tunings = map_action_to_pid_tunings(action)

        episode = supervised_plant_control.episode(setpoints, pid_tunings)
        save_and_plot_episode(timestamp_utc, episode)
        reward = evaluate(episode)

        print(f"action {action}/{pid_tunings} yielded reward {reward}")

