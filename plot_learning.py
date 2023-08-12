#!/usr/bin/env python
#
# A script to ingest a bunch of episodes and plot how the PID parameters
# progressed over time. This gives some idea on how learning progresses.
#
import sys
import pandas as pd
import matplotlib.pyplot as plt

from episodes import COL_TIME, COL_KP, COL_KI, COL_KD, COL_BENCHMARK, COL_ERROR


COL_KP_END = 'applied proportional gain $K_p$'
COL_KI_END = 'applied integral gain $K_i$'
COL_KD_END = 'applied derivative gain $K_d$'

LEARNING_COLUMNS = [
    COL_TIME,
    COL_KP, COL_KI, COL_KD,             # plotted in red: the start values
    COL_KP_END, COL_KI_END, COL_KD_END, # plotted in blue, the end values
    COL_BENCHMARK, COL_ERROR            # the sum of the squared error
]
learning = pd.DataFrame(columns=LEARNING_COLUMNS)

files = sys.argv[1:]
files.sort()
t = 0 # XXX
for file in files:
    episode = pd.read_parquet(file)

    first_step = episode.iloc[0]
    last_step = episode.iloc[-1]
    cumulative_error = (episode[COL_ERROR]**2).sum()

    # XXX that 9.0...
    episode_summary = [t,
                       first_step[COL_KP], first_step[COL_KI], first_step[COL_KD],
                       last_step[COL_KP], last_step[COL_KI], last_step[COL_KD],
                       9.0, cumulative_error]
    learning.loc[len(learning)] = episode_summary
    t = t+1

fig, axes = plt.subplot_mosaic("PPP;III;DDD;xyz", figsize=(15,10))

axes['P'].plot(learning[COL_TIME], learning[COL_KP],     color='r', linestyle=':', label='proposed ' + COL_KP)
axes['P'].plot(learning[COL_TIME], learning[COL_KP_END], color='b', label=COL_KP_END)
axes['P'].legend(loc='upper right')

axes['I'].plot(learning[COL_TIME], learning[COL_KI],     color='r', linestyle=':', label='proposed ' +COL_KI)
axes['I'].plot(learning[COL_TIME], learning[COL_KI_END], color='b', label=COL_KI_END)
axes['I'].legend(loc='upper right')

axes['D'].plot(learning[COL_TIME], learning[COL_KD],     color='r', linestyle=':', label='proposed ' +COL_KD)
axes['D'].plot(learning[COL_TIME], learning[COL_KD_END], color='b',                label=COL_KD_END)
axes['D'].legend(loc='upper right')

axes['x'].scatter(learning[COL_KP], learning[COL_KI],         color='r', alpha=0.2, label='proposed')
axes['x'].scatter(learning[COL_KP_END], learning[COL_KI_END], color='b',            label='applied')
axes['x'].set_xlabel(COL_KP)
axes['x'].set_ylabel(COL_KI)

axes['y'].scatter(learning[COL_KI], learning[COL_KD], color='r', alpha=0.2, label='proposed')
axes['y'].scatter(learning[COL_KI_END], learning[COL_KD_END], color='b',            label='applied')
axes['y'].set_xlabel(COL_KI)
axes['y'].set_ylabel(COL_KD)

axes['z'].scatter(learning[COL_KD], learning[COL_KP], color='r', alpha=0.2, label='proposed')
axes['z'].scatter(learning[COL_KD_END], learning[COL_KP_END], color='b',            label='applied')
axes['z'].set_xlabel(COL_KD)
axes['z'].set_ylabel(COL_KP)

plt.savefig("learning.png")
plt.close(fig)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(learning[COL_KP], learning[COL_KI], learning[COL_KD],             color='r', alpha=0.2, label='proposed')
ax.scatter(learning[COL_KP_END], learning[COL_KI_END], learning[COL_KD_END], color='b',            label='applied')

ax.set_xlabel(COL_KP)
ax.set_ylabel(COL_KI)
ax.set_zlabel(COL_KD)

ax.legend()

plt.savefig('learning-3d.png')
plt.close(fig)

# plot the cumulative error for all episodes
# add benchmark error to episode dataframe
# plot the benchmark error too
# add timestamp to title in episode plot


