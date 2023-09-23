#!/usr/bin/env python
#
# A script to ingest a bunch of episodes and plot how the PID parameters
# progressed over time. This gives some idea on how learning progresses.
#
import re
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from episodes import COL_TIME, COL_KP, COL_KI, COL_KD, COL_BENCHMARK, COL_ERROR, COL_STATE, STATE_NORMAL, STATE_FALLBACK


COL_KP_END = 'applied proportional gain $K_p$'
COL_KI_END = 'applied integral gain $K_i$'
COL_KD_END = 'applied derivative gain $K_d$'

LEARNING_COLUMNS = [
    COL_TIME,
    COL_KP, COL_KI, COL_KD,             # plotted in red: the start values
    COL_KP_END, COL_KI_END, COL_KD_END, # plotted in blue, the end values
    COL_BENCHMARK, COL_ERROR,           # the sum of the squared error
    COL_STATE                           # the final state of the episode
]
learning = pd.DataFrame(columns=LEARNING_COLUMNS)

files = sys.argv[1:]
files.sort()
for file in files:
    episode = pd.read_parquet(file)

    t_string = re.sub('.*/', '', file)
    t_string = re.sub('\..*', '', t_string)
    t = datetime.strptime(t_string, '%Y-%m-%dT%H%M%S')

    first_step = episode.iloc[0]
    last_step = episode.iloc[-1]
    cumulative_error = (episode[COL_ERROR]**2).sum()

    episode_summary = [t,
                       first_step[COL_KP], first_step[COL_KI], first_step[COL_KD],
                       last_step[COL_KP], last_step[COL_KI], last_step[COL_KD],
                       first_step[COL_BENCHMARK], cumulative_error,
                       last_step[COL_STATE]]
    learning.loc[len(learning)] = episode_summary

plt.rcParams['lines.linewidth'] = 0.8
fig, axes = plt.subplot_mosaic("EEE;PPP;III;DDD;xyz;klm;uvw", figsize=(15,15))

axes['E'].plot(learning[COL_TIME], learning[COL_ERROR],     color='orange',    label='episode error $RR_T$')
axes['E'].plot(learning[COL_TIME], learning[COL_BENCHMARK], color='lightgrey', label=COL_BENCHMARK)
axes['E'].set_ylim((0, first_step[COL_BENCHMARK] * 4))
axes['E'].legend(loc='upper left')

# ---

axes['P'].plot(learning[COL_TIME], learning[COL_KP],     color='r', linestyle=':', label='proposed ' + COL_KP)
axes['P'].plot(learning[COL_TIME], learning[COL_KP_END], color='b', label=COL_KP_END)
axes['P'].legend(loc='upper left')

axes['I'].plot(learning[COL_TIME], learning[COL_KI],     color='r', linestyle=':', label='proposed ' +COL_KI)
axes['I'].plot(learning[COL_TIME], learning[COL_KI_END], color='b', label=COL_KI_END)
axes['I'].legend(loc='upper left')

axes['D'].plot(learning[COL_TIME], learning[COL_KD],     color='r', linestyle=':', label='proposed ' +COL_KD)
axes['D'].plot(learning[COL_TIME], learning[COL_KD_END], color='b',                label=COL_KD_END)
axes['D'].legend(loc='upper left')

# ---

only_proposed = learning.loc[learning[COL_STATE] == STATE_FALLBACK]
only_applied  = learning.loc[learning[COL_STATE] == STATE_NORMAL]

axes['x'].scatter(only_proposed[COL_KP], only_proposed[COL_KI],       color='r', alpha=0.2)
axes['x'].scatter(only_applied[COL_KP_END], only_applied[COL_KI_END], color='b')
axes['x'].set_xlabel(COL_KP)
axes['x'].set_ylabel(COL_KI)

axes['y'].scatter(only_proposed[COL_KI], only_proposed[COL_KD],       color='r', alpha=0.2)
axes['y'].scatter(only_applied[COL_KI_END], only_applied[COL_KD_END], color='b')
axes['y'].set_xlabel(COL_KI)
axes['y'].set_ylabel(COL_KD)

axes['z'].scatter(only_proposed[COL_KD], only_proposed[COL_KP],       color='r', alpha=0.2)
axes['z'].scatter(only_applied[COL_KD_END], only_applied[COL_KP_END], color='b')
axes['z'].set_xlabel(COL_KD)
axes['z'].set_ylabel(COL_KP)

# ---

min_p = only_applied[COL_KP_END].min() * 0.9
max_p = only_applied[COL_KP_END].max() * 1.1
min_i = only_applied[COL_KI_END].min() * 0.9
max_i = only_applied[COL_KI_END].max() * 1.1
min_d = only_applied[COL_KD_END].min() * 0.9
max_d = only_applied[COL_KD_END].max() * 1.1
min_e = only_applied[COL_ERROR].min()  * 0.9
max_e = only_applied[COL_ERROR].max()  * 1.1

axes['k'].scatter(only_proposed[COL_KP_END], only_proposed[COL_KI_END], color='r', alpha=0.2)
axes['k'].scatter(only_applied[COL_KP_END], only_applied[COL_KI_END],   color='b')
if len(only_applied) > 1:
    axes['k'].plot(np.unique(only_applied[COL_KP_END]), np.poly1d(np.polyfit(only_applied[COL_KP_END], only_applied[COL_KI_END], 1))(np.unique(only_applied[COL_KP_END])), color='g')
axes['k'].set_xlim((min_p, max_p))
axes['k'].set_ylim((min_i, max_i))
axes['k'].set_ylabel(COL_KI)

axes['l'].scatter(only_proposed[COL_KI_END], only_proposed[COL_KD_END], color='r', alpha=0.2)
axes['l'].scatter(only_applied[COL_KI_END], only_applied[COL_KD_END],   color='b')
if len(only_applied) > 1:
    axes['l'].plot(np.unique(only_applied[COL_KI_END]), np.poly1d(np.polyfit(only_applied[COL_KI_END], only_applied[COL_KD_END], 1))(np.unique(only_applied[COL_KI_END])), color='g')
axes['l'].set_xlim((min_i, max_i))
axes['l'].set_ylim((min_d, max_d))
axes['l'].set_ylabel(COL_KD)

axes['m'].scatter(only_proposed[COL_KD_END], only_proposed[COL_KP_END], color='r', alpha=0.2)
axes['m'].scatter(only_applied[COL_KD_END], only_applied[COL_KP_END],   color='b')
if len(only_applied) > 1:
    axes['m'].plot(np.unique(only_applied[COL_KD_END]), np.poly1d(np.polyfit(only_applied[COL_KD_END], only_applied[COL_KP_END], 1))(np.unique(only_applied[COL_KD_END])), color='g')
axes['m'].set_xlim((min_d, max_d))
axes['m'].set_ylim((min_p, max_p))
axes['m'].set_ylabel(COL_KP)

# ---

axes['u'].scatter(only_proposed[COL_KP_END], only_proposed[COL_ERROR], color='r', alpha=0.2)
axes['u'].scatter(only_applied[COL_KP_END], only_applied[COL_ERROR],   color='b')
min_err_kp = only_applied[COL_KP_END][only_applied[COL_ERROR].idxmin()]
axes['u'].axvline(min_err_kp, color='g', label=f"$K_p$: {min_err_kp}")
axes['u'].set_xlim((min_p, max_p))
axes['u'].set_xlabel(COL_KP)
axes['u'].set_ylim((min_e, max_e))
axes['u'].set_ylabel(COL_ERROR)
axes['u'].legend(loc='upper left')

axes['v'].scatter(only_proposed[COL_KI_END], only_proposed[COL_ERROR], color='r', alpha=0.2)
axes['v'].scatter(only_applied[COL_KI_END], only_applied[COL_ERROR],   color='b')
min_err_ki = only_applied[COL_KI_END][only_applied[COL_ERROR].idxmin()]
axes['v'].axvline(min_err_ki, color='g', label=f"$K_i$: {min_err_ki}")
axes['v'].set_xlim((min_i, max_i))
axes['v'].set_ylim((min_e, max_e))
axes['v'].set_xlabel(COL_KI)
axes['v'].legend(loc='upper left')

axes['w'].scatter(only_proposed[COL_KD_END], only_proposed[COL_ERROR], color='r', alpha=0.2)
axes['w'].scatter(only_applied[COL_KD_END], only_applied[COL_ERROR],   color='b')
min_err_kd = only_applied[COL_KD_END][only_applied[COL_ERROR].idxmin()]
axes['w'].axvline(min_err_kd, color='g', label=f"$K_d$: {min_err_kd}")
axes['w'].set_xlim((min_d, max_d))
axes['w'].set_ylim((min_e, max_e))
axes['w'].set_xlabel(COL_KD)
axes['w'].legend(loc='upper left')

# ---

plt.savefig("learning.png")
plt.close(fig)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(only_proposed[COL_KP], only_proposed[COL_KI], only_proposed[COL_KD],          color='r', alpha=0.2, label='proposed')
ax.scatter(only_applied[COL_KP_END], only_applied[COL_KI_END], only_applied[COL_KD_END], color='b',            label='applied')

ax.set_xlabel(COL_KP)
ax.set_ylabel(COL_KI)
ax.set_zlabel(COL_KD)

ax.legend(loc="upper left")

plt.savefig('learning-3d.png')
plt.close(fig)

