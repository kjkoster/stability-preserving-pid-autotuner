#!/usr/bin/env python
#
# This is the auto-tuner that generates new PID parameters. The auto-tuner is
# kept separate from the plant control. It works solely on the data that comes
# out of the running, supervised plant control and proposed alternative PID
# tunings for it.
#
import numpy as np
from ddpg_torch import Agent
from datetime import datetime

from episodes import T, SAMPLE_RATE, EPISODE_LENGTH, COL_CONTROL_VARIABLE, COL_PROCESS_VARIABLE, COL_ERROR, save_and_plot_episode
from ddpg_torch import OUActionNoise
from plant_control import PlantControl
from supervised_plant_control import SupervisedPlantControl

IS_HARDWARE = False

PID_TUNINGS = (50.0, 0.001, 0.1)
SET_POINT = 23.0

BENCHMARK_ERROR = 15.0
FALLBACK_PID_TUNINGS = (20.0, 0.1, 0.01)

MAP_GAINS = [500.0, 50.0, 5.0]

N_ACTIONS = 3
BATCH_SIZE = 64

def map_action_to_pid_tunings(action):
    return (action[0] * MAP_GAINS[0], action[1] * MAP_GAINS[1], action[2] * MAP_GAINS[2])


noise = OUActionNoise(np.zeros(N_ACTIONS))
def noisy_fall_back_tunings():
    noisy_gains = FALLBACK_PID_TUNINGS + noise()
    return [noisy_gains[0] / MAP_GAINS[0], noisy_gains[1] / MAP_GAINS[1], noisy_gains[2] / MAP_GAINS[2]], \
           (noisy_gains[0], noisy_gains[1], noisy_gains[2])


class RandomAgent(object):
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def choose_action(self):
        return np.random.rand(self.n_actions)


def evaluate(episode):
    if len(episode) < 12:
        observed_data = episode[[COL_CONTROL_VARIABLE, COL_PROCESS_VARIABLE]].reindex(range(12)).fillna(0.0)
    else:
        observed_data = episode.iloc[np.linspace(0, len(episode), 12, endpoint=False)][[COL_CONTROL_VARIABLE, COL_PROCESS_VARIABLE]]
    observed_data = observed_data.values.flatten().tolist()

    error = -(episode[COL_ERROR]**2).sum()

    return observed_data, error

#
# The main driver. Create a supervised plan control and an agent. Prime the
# learning process, then start running episodes and evaluating these with the
# auto-tuner.
#
if __name__ == "__main__":
    plant_control = PlantControl(IS_HARDWARE)
    plant_control.set_pid_tunings(FALLBACK_PID_TUNINGS, "program starts")

    supervised_plant_control = SupervisedPlantControl(plant_control, BENCHMARK_ERROR, FALLBACK_PID_TUNINGS)

    random_agent = RandomAgent(N_ACTIONS)
    agent = Agent(alpha=0.00005, beta=0.0005, input_dims=[24], tau=0.001,
                  batch_size=BATCH_SIZE, layer1_size=400, layer2_size=300, n_actions=N_ACTIONS, max_size=1_000_000)

    print("generating priming step...")
    episode, _ = supervised_plant_control.step(SET_POINT)
    observation, _ = evaluate(episode)

    episode_nr = 0
    while True:
        episode_nr += 1
        if episode_nr <= 250:
            action, pid_tunings = noisy_fall_back_tunings()
        elif episode_nr <= 500:
            action = random_agent.choose_action()
            pid_tunings = map_action_to_pid_tunings(action)
        else:
            action = agent.choose_action(observation)
            pid_tunings = map_action_to_pid_tunings(action)

        episode, done = supervised_plant_control.step(SET_POINT)
        if done:
            timestamp_utc = datetime.utcnow()
            print(f"saving episode {timestamp_utc.isoformat()}...")
            save_and_plot_episode(timestamp_utc, episode)

        new_state, reward = evaluate(episode)

        agent.remember(observation, action, reward, new_state, done)
        agent.learn()

        observation = new_state

