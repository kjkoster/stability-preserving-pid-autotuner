# Iterations of the Project

This page documents some of the findings and iterations that this project has gone through. If you find it interesting to know how the project developed over time, this page is for you.

The topics are organised in chronological order, in a sort of logbook format.

---
## 2023-08-12: Tanh versus Relu

The agent was producing negative values for the PID gain values. As PID controllers expect all parameters to be of the same sign (either all positive or all negative), that was a problem. As a stop-gap measure I mapped all negative values to 0. I believe that led to the many, many 0 values that the agent ended up setting. Consider the learning graph below:

<p align="center" width="100%">
    <img width="50%" src="../images/tanh-problem.png"> 
</p>

In that graph, we see that the agent predominantly chose 0 values for gains, sometimes even 0 for all three gains. I did not test for this explicitly, but intuitively I would say that this will happen when half the action space is mapped onto 0.

Based on these observations, I chose to switch from Tanh activation to ReLu activation. The advantage of ReLu is that it only produces positive values.

At first, I thought it would be enough to just switch the activation function to ReLu, but that turned out to be wrong. The policies were being initialised with `-f1`, `-f2` and `-f3` to negative values too. After fixing that, I still got negative action values.

<p align="center" width="100%">
    <img width="50%" src="../images/relu-goes-negative.png"> 
</p>

I traced that to the noise generator. While the noise is _added_ to the signal, negative noise on small signal values may still cause the sum to be negative. After taking the absolute value of the noise this problem was solved too and I only got positive values.

So to switch the agent to ReLu activation you need all three of setting the final activation to ReLu, initialise the agent only with positive values and only generate positive noise values. After this switch, the agent no longer tried to bring all values down to 0. This last set of graphs shows the agent happily meandering through the search space. this shown best by the small, red-and-blue scatter plots that show the three gains each plotted against another.

<p align="center" width="100%">
    <img width="50%" src="../images/relu-meander-properly.png"> 
</p>

