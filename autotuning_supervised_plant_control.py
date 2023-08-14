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
from plant_control import PlantControl
from supervised_plant_control import SupervisedPlantControl

IS_HARDWARE = False

PID_TUNINGS = (50.0, 0.001, 0.1)
SET_POINT = 23.0

BENCHMARK_ERROR = 15.0
FALLBACK_PID_TUNINGS = (20.0, 0.1, 0.01)

def evaluate(episode):
    return episode.iloc[range(0, T, int(T/12))][[COL_CONTROL_VARIABLE, COL_PROCESS_VARIABLE]].values.reshape(24, 1).flatten().tolist() , \
           -(episode[COL_ERROR]**2).sum()


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
    agent = Agent(alpha=0.00005, beta=0.0005, input_dims=[24], tau=0.001,
                  batch_size=64, layer1_size=400, layer2_size=300, n_actions=3, max_size=1_000_000)

    # knowing nothing, we just start with the fallback tunings
    pid_tunings = FALLBACK_PID_TUNINGS

    # run a first episode, we need a first episode to prime the learning cycle
    timestamp_utc = datetime.utcnow()
    print(f"generating priming episode {timestamp_utc.isoformat()}...")
    episode = supervised_plant_control.episode(setpoints, pid_tunings)
    save_and_plot_episode(timestamp_utc, episode)
    observation, _ = evaluate(episode)

    while True:
        timestamp_utc = datetime.utcnow()
        print(f"generating episode {timestamp_utc.isoformat()}...")

        action = agent.choose_action(observation)
        pid_tunings = (max(0, action[0]) * 500.0, max(0, action[1]) * 50.0, max(0, action[2]) * 5.0)

        episode = supervised_plant_control.episode(setpoints, pid_tunings)
        save_and_plot_episode(timestamp_utc, episode)
        new_state, reward = evaluate(episode)

        print(f"action {action}/{pid_tunings} yielded reward {reward}")
        # we treat each episode as done, otherwise the system won't learn at all
        agent.remember(observation, action, reward, new_state, True)
        agent.learn()

        observation = new_state

