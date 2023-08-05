#
# The core functions.
#

import numpy as np
import matplotlib.pyplot as plt

#
# Using $\LaTeX$ in the variable names works well for rendering tables and for
# MatPlotlib, but they break Pandas' built-in plotting due to the use of
# backslashes in the column names. We use MatPlotlib, so that should not be an
# issue.
#

COL_TIME = 'time $sec$'
COL_SETPOINT = 'setpoint $^oC\ r(t)$'
COL_ERROR = 'error $^oC\ e(t)$'
COL_KP = 'proportional gain $K_p$'
COL_KI = 'integral gain $K_i$'
COL_KD = 'derivative gain $K_d$'
COL_INTERNAL_PROPORTIONAL = '(pid internal) proportional'
COL_INTERNAL_INTEGRAL = '(pid internal) integral'
COL_INTERNAL_DERIVATIVE = '(pid internal) derivative'
COL_CONTROL_VARIABLE = 'control variable $\%\ u(t)$'
COL_CONTROL_VARIABLE_UNCAPPED = 'control variable uncapped $\%\ u(t)$'
COL_PROCESS_VARIABLE = 'process variable $^oC\ y(t)$'
COL_DISTURBANCE_CONTROL_VARIABLE = 'disturbance control variable $\%\ u2(t)$'
COL_SECONDARY_PROCESS_VARIABLE = 'secondary process variable $^oC\ y2(t)$'
COL_STATE = 'state'

#
# What a Panda's data frame looks like for each episode.
#

EPISODE_COLUMNS = [COL_TIME, COL_SETPOINT,
                   COL_KP, COL_KI, COL_KD,
                   COL_INTERNAL_PROPORTIONAL, COL_INTERNAL_INTEGRAL, COL_INTERNAL_DERIVATIVE,
                   COL_ERROR, COL_CONTROL_VARIABLE_UNCAPPED, COL_CONTROL_VARIABLE, COL_PROCESS_VARIABLE,
                   COL_DISTURBANCE_CONTROL_VARIABLE, COL_SECONDARY_PROCESS_VARIABLE,
                   COL_STATE]

#
# These are the supervisor states.
#

STATE_NORMAL = 0
STATE_FALLBACK = 1

T = 300                          # nominal episodes are 5 minutes, or 300 seconds.
SAMPLE_RATE = 2 # Hz             # the hardware samples take varying times, but anything under ~2.5 Hz looks safe
EPISODE_LENGTH = T * SAMPLE_RATE # Multiply by sample rate to get the episode and data frame size.

#
# To get good insights in how the system behaves, we define a plotting function.
# This function takes the output of a simulation and breaks it into a few
# graphs. We combine in the first plot all temperatures related values, such as
# $r(t)$ and $y(t)$. The process variable is shown in the second plot, as it is
# a percentage. Finally, we take a sneak peek at internal state of the PID
# controller.
#
def plot_episode(_df, episode_plot=None):
    to_fallback = np.searchsorted(_df[COL_STATE], STATE_FALLBACK) / SAMPLE_RATE

    plt.rcParams['lines.linewidth'] = 0.8
    fig, axes = plt.subplot_mosaic("TTT;TTT;HHH;HHH;PID", figsize=(15,10))

    # a mix of string concatenations because LaTeX confuses Python formatters
    error_label = COL_ERROR + ', $\sum_{t=0}^{T}e^2(t) = ' + f"{(_df[COL_ERROR]**2).sum():.1f}" + '$'

    axes['T'].plot(_df[COL_TIME], _df[COL_SETPOINT],                   'k',  label=COL_SETPOINT)
    axes['T'].plot(_df[COL_TIME], _df[COL_PROCESS_VARIABLE],           'b',  label=COL_PROCESS_VARIABLE)
    axes['T'].plot(_df[COL_TIME], _df[COL_SECONDARY_PROCESS_VARIABLE], 'g:', label=COL_SECONDARY_PROCESS_VARIABLE)
    axes['T'].plot(_df[COL_TIME], _df[COL_ERROR],                      'r',  label=error_label)
    if to_fallback < T:
        axes['T'].axvspan(to_fallback, T, facecolor='peachpuff', alpha=0.3)
    axes['T'].set_ylabel(r'temperature $(^oC)$')
    axes['T'].legend(loc='upper right')
    
    axes['H'].axhline(y=0.0,   color='grey', linestyle=':', alpha=0.5)
    axes['H'].axhline(y=100.0, color='grey', linestyle=':', alpha=0.5)
    axes['H'].plot(_df[COL_TIME], _df[COL_CONTROL_VARIABLE_UNCAPPED],    'r--', label=COL_CONTROL_VARIABLE_UNCAPPED)
    axes['H'].plot(_df[COL_TIME], _df[COL_CONTROL_VARIABLE],             'b',   label=COL_CONTROL_VARIABLE)
    axes['H'].plot(_df[COL_TIME], _df[COL_DISTURBANCE_CONTROL_VARIABLE], 'g:',  label=COL_DISTURBANCE_CONTROL_VARIABLE)
    if to_fallback < T:
        axes['H'].axvspan(to_fallback, T, facecolor='peachpuff', alpha=0.3)
    axes['H'].set_ylabel('heater $(\%)$')
    axes['H'].set_ylim((-50.0, 150.0))
    axes['H'].legend(loc='upper right')
    
    axes['P'].axhline(y=0.0, color='grey', linestyle=':', alpha=0.5)
    axes['P'].plot(_df[COL_TIME], _df[COL_INTERNAL_PROPORTIONAL], label=COL_INTERNAL_PROPORTIONAL)
    if to_fallback < T:
        axes['P'].axvspan(to_fallback, T, facecolor='peachpuff', alpha=0.3)
    axes['P'].legend(loc='upper right')

    axes['I'].axhline(y=0.0, color='grey', linestyle=':', alpha=0.5)
    axes['I'].plot(_df[COL_TIME], _df[COL_INTERNAL_INTEGRAL],     label=COL_INTERNAL_INTEGRAL)
    if to_fallback < T:
        axes['I'].axvspan(to_fallback, T, facecolor='peachpuff', alpha=0.3)
    axes['I'].legend(loc='upper right')

    axes['D'].axhline(y=0.0, color='grey', linestyle=':', alpha=0.5)
    axes['D'].plot(_df[COL_TIME], _df[COL_INTERNAL_DERIVATIVE],   label=COL_INTERNAL_DERIVATIVE)
    if to_fallback < T:
        axes['D'].axvspan(to_fallback, T, facecolor='peachpuff', alpha=0.3)
    axes['D'].legend(loc='upper right')

    if episode_plot is None:
        fig.show();
    else:
        plt.savefig(episode_plot)
        plt.close(fig)

