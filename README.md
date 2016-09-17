# Input Convex Neural Networks (ICNNs)

This repository is by [Brandon Amos](http://bamos.github.io),
[Leonard Xu](https://github.com/Leonard-Xu),
and [J. Zico Kolter](http://zicokolter.com)
and contains the TensorFlow source code to reproduce the
experiments in our paper
[Input Convex Neural Networks](TODO).

If you find this repository helpful in your publications,
please consider citing our paper.

```
TODO: Citation
```

## Synthetic Classification

TODO

## Image Completion

[TODO: Animation of completions]

### Set Up
[TODO: Download Olivetti, TensorFlow]

### [completion/icnn_ebundle.py](/completion/icnn_ebundle.py)

Code to train a completion PICNN with the entropy bundle
method on the Olivetti faces dataset.

[TODO: Example of running]

### [completion/icnn_backopt.py](/completion/icnn_backopt.py)

Code to train a completion PICNN with back
optimization on the Olivetti faces dataset.

[TODO: Example of running]

## Reinforcement Learning

![image](/RL/misc/pendulum.gif)
![image](/RL/misc/reacher.gif)
![image](/RL/misc/halfcheetah.gif)

###Dependency

- Tensorflow r10
- OpenAI Gym + Mujoco
- numpy

###Set Up
**Training**

Example

```
python main.py --model ICNN --env InvertedPendulum-v1 --outdir output \
  --total 100000 --train 100 --test 1 --tfseed 0 --npseed 0 --gymseed 0
```

Use `--model` to select a model from `[DDPG, NAF, ICNN]`.

Use `--env` to select a task. [TaskList](https://gym.openai.com/envs#mujoco)

Check all parameters using `python main.py -h`.

**Output**

Tensorboard summary is on by default. Use `--summary False` to turn it off. Tensorboard summary includes (1) average Q value, (2) loss function, and (3) average reward for each training minibatch. 

Use log.txt to log testing total rewards. Each line is `[training_timesteps]	[testing_episode_total_reward]`. 

###Acknowledgement
Part of the code is modified from Repo [SimonRamstedt/ddpg](https://github.com/SimonRamstedt/ddpg). 
