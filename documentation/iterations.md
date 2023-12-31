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

---
## 2023-08-20: Priming the Memory

With the agent happily plodding through the action and observation spaces, I found that the agent would wander very far away from what is likely to be a useful area to explore. I deliberately set the bounds for each gain quite wide, so that I would be able to see how well the agent would zoom in on where the optimal policies might lie. If you look at the last graph of the Tanh/ReLu logbook entry, you will see that there are only a few blue spikes where the agent found useful values for the PID gains. However, it then goes off to explore further out, instead of exploring around the acceptable values.

We are not training some generic agent and we should use knowledge about the problem domain to improve our system any way we can. We are trying to optimize a PID controller that has been hand-tuned to be reasonably stable already. My intuition tells me that stable values for the PID gains should be clustered in the vicinity of the hand-tuned values.

To validate the intiution, we will run for a while wth completely random actions. That should reveal how the clustering of the useful action space looks like. To do that, we have `random-but-stable-pid-autotuner.py`. This implements an agent that makes no attempt to learn or optimise. All it does is generate random values for the PID gains, so that we can see if there is clustering in the action space or not.

<p align="center" width="100%">
    <img width="50%" src="../images/priming-fully-random.png">
</p>

As an aside: I really like the supervisor model that the paper proposed. It makes experimenting with a live system quite doable.

From the graphs of the completely random agent, it is quite clear that effective parameters are clustered around the human-provided fall-back parameter set. It hit a few values that might work, but these fall into the same, tiny cluster.

Comparing the DDPG graphs with the random graphs shows that DDPG meanders around in the search space in smaller steps. This has been talked about in the various video's and papers on the stopic, but it is nice to see it realy do so. However, this random graph shows that it covers the action space much more evenly. The DDPG based agents seem to leave a lot of the search space unexplored.

Lets combine these things and prime the agent's replay buffer with information that should make it learn better. In most RL projects, we try to reduce biasing the system towards a certain solution, but here we take the opposite approach. We know roughly where the solution lies and want to put a big roadsign pointing to that cluster. We first run the known stable PID tunings for a while and then run with random tunings for a bit. The idea is that this shows the reinforcement learning agent where the optimal cluster is and that the rest of the search space is mostly a terrible choice.

The left of the figures below show a run where we start with 250 samples in the cluster around the fall-back parameter set. Each episode, we add a bit of noise to the fall-back gains and use that as our parameter pack. Then we run 250 episodes woth fully random PID tunings, as shown in the right of the graphs below. These are both graphs from the same run, just each snapshot taken at a different run-time length.
After these 500 episodes, the agent is started as normal.

<p align="center" width="100%">
    <img width="30%" src="../images/priming-first-prime.png">
    <img width="30%" src="../images/priming-after-priming.png">
</p>

Perhaps I should have made the spread of the noise wider.

With the priming done, the agent is once again set free. It does give the impresstion of learning properly this time.. This is shown as the red splotch around the blue, probably-optimal cluster. The agent seems prone to over-estimation, which I am told is a weakness of DDPG.

<p align="center" width="100%">
    <img width="50%" src="../images/priming-done.png">
</p>

Keep in mind that the number of cycles I run is very low, due to me programming the whole thing as a wall-clock time simulation. My conclusions are bordering wishful thinking.

Overall, I am quite content with this priming idea, and I will keep it in.

## 2023-09-01: Smaller Search Space

Just to see what happens, I reduced the search space for the agent to about a quarter of its original size. You can see the comparison in the graphs below. On the left, I have the original search space. On the right I have the smaller search space.

<p align="center" width="100%">
    <img width="30%" src="../images/priming-done.png">
    <img width="30%" src="../images/priming-smaller-searchspace.png">
</p>

While it may look like this is an improvement, it really is just the same result: the agent is wandering around the search space a little and accidentally blunders into working parameter sets. When the search space is smaller, the accidents are more frequent.

If anything, this shows how little of the search space is actually explored by the agent in the roughy 1000 episodes that I run this system for. If I want any sort of predictability, the agent would probably need to have seen most of the search space.

If you look at the bottom graphs, you can see that a simple regression would probably have the agent zoom in on the optimal parameter set in no time at all. If you decide to half-automate tuning, just run your controller under the supervisor, run the random agent for a while to determine the cluster of usable values. Next restrict the search space to the cluster and then re-run the random agent. Finally do a regression and that is your optimal value. Repeat every month or two and you don't need machine learning at all.

I will stick with the larger search space, just because it is more difficult for the agent to achieve good results. On to the next experiment.

## 2023-09-23: Stepwise Learning

Up until now I only ran the learning cycle at the end of each 5-minute episode. This means that the agent cannot learn from the 299 intermediate episode steps. This is not how all the other implementations work. In this iteration I reworked the code to make the agent learn from the intermediate episodes too.

The DDPG agent is set up to work in this fashion. The learning operation actually bahaves differently with the episode is marked as `done`.

