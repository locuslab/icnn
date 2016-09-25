# Input Convex Neural Networks (ICNNs)

This repository is by [Brandon Amos](http://bamos.github.io),
[Leonard Xu](https://github.com/Leonard-Xu),
and [J. Zico Kolter](http://zicokolter.com)
and contains the TensorFlow source code to reproduce the
experiments in our paper
[Input Convex Neural Networks](TODO).

![](/RL/misc/pendulum.gif)
![](/RL/misc/reacher.gif)
![](/RL/misc/halfcheetah.gif)
![](/images/completion.png)

If you find this repository helpful in your publications,
please consider citing our paper.

```
TODO: Citation
```

## General Purpose Libraries

```
lib
└── bundle_entropy.py - Optimize a function over the [0,1] box with the bundle entropy method.
```

## Synthetic Classification

![](/images/synthetic.png)

```
synthetic-cls
├── icnn.py - Main script.
├── legend.py - Create a figure of just the legend.
├── make-tile.sh - Make the tile of images.
└── run.sh - Run all experiments on 4 GPUs.
```

## Multi-Label Classification

```
multi-label-cls
├── bibsonomy.py - Loads the Bibsonomy datasets.
├── ebundle-vs-gd.py - Compare ebundle and gradient descent.
├── ff.py - Train a feed-forward net baseline.
├── icnn_ebundle.py - Train an ICNN with the bundle entropy method.
├── icnn.back.py - Train an ICNN with gradient descent and back differentiation.
└── icnn.plot.py - Plot the results from any multi-label cls experiment.
```

## Image Completion

![](/images/completion.png)

```
completion
├── icnn.back.py - Train an ICNN with gradient descent and back differentiation.
├── icnn_ebundle.py - Train an ICNN with the bundle entropy method.
├── icnn.plot.py - Plot the results from any image completion experiment.
└── olivetti.py - Loads the Olivetti faces dataset.
```

## Reinforcement Learning

![](/RL/misc/pendulum.gif)
![](/RL/misc/reacher.gif)
![](/RL/misc/halfcheetah.gif)

###Dependency

- Tensorflow r10
- OpenAI Gym + Mujoco
- numpy

###Set Up
**Training**

Example

```
python src/main.py --model ICNN --env InvertedPendulum-v1 --outdir output \
  --total 100000 --train 100 --test 1 --tfseed 0 --npseed 0 --gymseed 0
```

Use `--model` to select a model from `[DDPG, NAF, ICNN]`.

Use `--env` to select a task. [TaskList](https://gym.openai.com/envs#mujoco)

Check all parameters using `python main.py -h`.

**Output**

Tensorboard summary is on by default. Use `--summary False` to turn it off. Tensorboard summary includes (1) average Q value, (2) loss function, and (3) average reward for each training minibatch.

Use log.txt to log testing total rewards. Each line is `[training_timesteps]	[testing_episode_total_reward]`.

**Settings**

To reproduce experiments, you can use [scripts](/RL/scripts/). Please run scripts in `RL` directory use `bash script/*.sh`. Outputs and figure will be saved at `RL/output/*`.


### Acknowledgment
The DDPG portions of the code are from
[SimonRamstedt/ddpg](https://github.com/SimonRamstedt/ddpg)
and remain under the original MIT license
by the original authors.

# Licensing

Unless otherwise stated, the source code is copyright
Carnegie Mellon University and licensed under the
[Apache 2.0 License](./LICENSE).
Portions from the following third party sources have
been modified and are included in this repository.
These portions are noted in the source files and are
copyright their respective authors with
the licenses listed.

Project | License
---|---|
| [SimonRamstedt/ddpg](https://github.com/SimonRamstedt/ddpg) | MIT |
