# Stability Preserving PID Autotuner

Industrial and marine systems use [PID](https://en.wikipedia.org/wiki/PID_controller)
controllers for a lot of things. They are simple to use and very effective. PID
controller tuning is an area that still leaves room for improvement. In many
cases, PIDs are quickly hand-tuned and then left to operate under what is likely
a suboptimal set of parameters. This makes tuning of PID controllers ripe for
automation.

As a system ages, its behaviour may change over time. Materials wear and
components may be swapped out for equivalent, but not identical, replacements.
In an ideal world, all PID controllers on systems would be periodically retuned
to compensate for changes in response of systems. If not periodically, then at
least they should be retuned whenever components are replaced. In practice this
rarely happens. Even the initial tuning is often done quickly and
conservatively. A PID controller with a fixed set of parameters is not equipped
to adapt to this.

Specifically in an environment where energy conservation is important, well
tuned PID controllers can help eek out the last few drops of performance.

This project explores a safe, stability-preserving, Reinforcement Learning (RL)
based automatic PID controller tuning mechanism. The work of this project is
heavily based on [Stability-preserving automatic tuning of PID control with reinforcement learning](https://comengsys.com/article/view/4601) by Ayub I. Lakhani, Myisha A. Chowdhury and Qiugang Lu,
which is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
It is also [available on Youtube](https://www.youtube.com/watch?v=ymodIJ7yMKo).
This work will be referred to as "the paper" in througout this project.

There are benefits to using reinforcement learning not to learn the control of a
system, but rather learn how to tune a PID controller to find the optimal PID
control for a system. PID controllers are well understood and mathematically
easy to explain. For systems in environments where human lives are dependent on
the good operation of systems, the verifiability of the operation of that system
is very important. Pure RL control would make the control system into a black
box. RL systems are not considered to be in the category of explainable machine
learning models. By limiting the scope of RL to PID tuning, the tuning process
may be a black box, while the resultant control system is still well understood
and explainable.

In an emergency when the reinforcement learning were to break down, humans can
still go in, take control and hand-tune the PID controller. This gives engineers
the option to maintain automatic control under partial systems failure.

Finally, reinformcement learning does not tire or get bored. It follows subtle
changes in systems response.

**Future work:**

* Use the changes in PID control to detect tiny changes in systems behaviour, possibly as early warning system for maintenance.
* http://brettbeauregard.com/blog/2011/04/improving-the-beginner%E2%80%99s-pid-tuning-changes/
* Explore how we might have two separate PID controllers: one that responds to setpoint changes and one that is good at tracking stable setpoint values.
* Make the processes queue based internally.
* Consider starting a fresh episode whenever the setpoint changes (significantly). That way, we have a predictable error form to work with in each episode, at the expense of having to disregard episodes that we cur short.

**Limitations:**
* systems with relatively few learning episodes (winches?) or where it is hard to measure the feedback.
* The paper assumes that setpoint is fixed, so this is for systems where the only disturbances are system wear and outside influences.

**TODO**:

groundwork

* background task to generate graphs (but watch the clock, we cannot have the system sleeping).
* set up an episode-generating server somewhere

training

* import phil's agent code
* document phil's code
* bolt the disk episodes onto the memory class
* add diagrams
* attributions etc

applying

* make a optimizer-supervisor class

More:

* buy a tclab or two (awaiting delivery)
* _after delivery_: Explore sample rate limits with the hardware version: https://tclab.readthedocs.io/en/latest/notebooks/02_Accessing_the_Temperature_Control_Laboratory.html#tclab-Sampling-Speed
* check time progression in both tclab and simple_pid. I think I am mixing real-time and sped-up times on these, which will screw up the integral and derivative calculations.
* Add consistent time speedup to the `PlantControl` class.
* decide: do I cut episodes short? That way the step response episodes are of better quality, since they will start at the setpoint change.
* add start values for the PID controller, otherwise we get breaks between episodes
* Do I keep the pattern of comparing running with totals?
* make time compression possible for simulated envronment, read up on how simple_pid does that.
* _after DDPG_ consider convolutions,

---
## Flawed Premise
Of course, the whole premise for this idea is flawed. The reason not to tune PID
controllers is in part lack of knowledge and in part lack of a real need,
finished off by the fact that developers choose predictability over performance.
Making a machine learning based autotuner solves none of these. If anything, the
relative novelty of machine learning for this application will drive developers
away from using it.

So this tuner either works fully automatically and invisibly in the background,
or it will never be used.

So yeah, there is that...

---
## Virtual Environment and Dependencies
We tried to lock down dependencies into a `requirements.txt` file, but not all
dependencies are trivial to install via the `pip` command. Notably, maintenance
of TCLab has stopped due to personal circumstances of the maintainer. The latest
`pip`-installable version is not compatible with the newer Python versions.
Thus, we install that package manually.

```bash
$ python3 -m venv venv
$ source venv/bin/activate
(venv) $ pip install https://github.com/jckantor/TCLab/archive/master.zip
(venv) $ pip install -r requirements.txt
```

With the virtual environment set up, you are now ready to run the code for this
project. If you don't use virtual environments a lot, don't forget to activate
it when you return to the project.

---
## Basic Plant Control
Before we worry about the complexity of supervision and automatic PID tuning, we
need an environment where we can control a plant. This script brings the control
components together into a working simulation. These just use a fixed set of PID
parameters.

The programming is cyclic, just like it would be on a
[PLC](https://en.wikipedia.org/wiki/Programmable_logic_controller), for example.
In fact, if you own a TCLab device, you can use this loop to control that. Much
as I like matrix processing and its efficiency, the matrix programming model
does not fit the continuous control loop that is common for live systems.

For now, the setpoint is just a fixed value of 23 $\celsius$.

```sh
(venv) $ python plant_control.py
^C
(venv) $ _
```

The program runs continuously. You can break out of it using `^C`.

If you own a TCLab device, edit the script to set `IS_HARDWARE` to `True` (it
defaults to `False`). That will make the control loop start controlling the
actual device.

---
## Autotuner
With the supervisor ready to take over in case the control loop becomes
unstable, we turn out attention to the auto tuning. As the paper has, we will
use [Deep Deterministic Policy Gradients (DDPG)](https://www.youtube.com/watch?v=6Yd5WnYls_Y).

The code is largely copied from our own [PyTorch DDPG Tutorial Implementation](https://github.com/kjkoster/ddpg-continuous-tutorial), which in turn is a 99% copy of [Reinforcement Learning in Continuous Action Spaces | DDPG Tutorial Pytorch](https://www.youtube.com/watch?v=6Yd5WnYls_Y)
by [Machine Learning with Phil](https://www.youtube.com/@MachineLearningwithPhil).

