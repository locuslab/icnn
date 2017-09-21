# Input Convex Neural Networks (ICNNs)

This repository is by [Brandon Amos](http://bamos.github.io),
[Leonard Xu](https://github.com/Leonard-Xu),
and [J. Zico Kolter](http://zicokolter.com)
and contains the TensorFlow source code to reproduce the
experiments in our ICML 2017 paper
[Input Convex Neural Networks](http://arxiv.org/abs/1609.07152).

![](/RL/misc/pendulum.gif)
![](/RL/misc/reacher.gif)
![](/RL/misc/halfcheetah.gif)
![](/images/completion.gif)

If you find this repository helpful in your publications,
please consider citing our paper.

```
@InProceedings{amos2017icnn,
  title = {Input Convex Neural Networks},
  author = {Brandon Amos and Lei Xu and J. Zico Kolter},
  booktitle = {Proceedings of the 34th International Conference on Machine Learning},
  pages = {146--155},
  year = {2017},
  volume = {70},
  series = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
}
```

## Setup and Dependencies

+ Python/numpy
+ TensorFlow (we used r10)
+ OpenAI Gym + Mujoco (for the RL experiments)

## Libraries

```
lib
└── bundle_entropy.py - Optimize a function over the [0,1] box with the bundle entropy method.
                        (Development is still in-progress and we are still
                        fixing some numerical issues here.)
```

## Synthetic Classification

This image shows FICNN (top) and PICNN (bottom) classification of synthetic
non-convex decision boundaries.

![](/images/synthetic.png)

```
synthetic-cls
├── icnn.py - Main script.
├── legend.py - Create a figure of just the legend.
├── make-tile.sh - Make the tile of images.
└── run.sh - Run all experiments on 4 GPUs.
```

## Multi-Label Classification

(These are currently slightly inconsistent with our paper
and we plan on synchronizing our paper and code.)

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

This image shows the test set completions on the Olivetti faces dataset over
the first few iterations of training a PICNN with the bundle entropy method
for 5 iterations.

![](/images/completion.gif)

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

### Training

From the [RL](/RL) directory, run a single experiment with:

```
python src/main.py --model ICNN --env InvertedPendulum-v1 --outdir output \
  --total 100000 --train 100 --test 1 --tfseed 0 --npseed 0 --gymseed 0
```

+ Use `--model` to select a model from `[DDPG, NAF, ICNN]`.
+ Use `--env` to select a task. [TaskList](https://gym.openai.com/envs#mujoco)
+ View all of the parameters with `python main.py -h`.

### Output

The TensorBoard summary is on by default. Use `--summary False` to
turn it off. The TensorBoard summary includes (1) average Q value, (2)
loss function, and (3) average reward for each training minibatch.

The testing total rewards are logged to log.txt.
Each line is `[training_timesteps]	[testing_episode_total_reward]`.

### Settings

To reproduce our experiments, run the scripts in the
[RL](/RL) directory.

### Acknowledgments
The DDPG portions of our RL code are from
Simon Ramstedt's
[SimonRamstedt/ddpg](https://github.com/SimonRamstedt/ddpg)
repository.

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
