import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('runs', 10, 'total runs')
flags.DEFINE_integer('total', 100000, 'total timesteps')
flags.DEFINE_integer('train', 1000, 'training timesteps between testing')
flags.DEFINE_string('data', '.', 'dir contains outputs of DDPG, NAF and ICNN')


folders = [
    [FLAGS.data + '/DDPG/%d' % i for i in xrange(FLAGS.runs)],
    [FLAGS.data + '/NAF/%d' % i for i in xrange(FLAGS.runs)],
    [FLAGS.data + '/ICNN/%d' % i for i in xrange(FLAGS.runs)],
]
names = [
    'DDPG',
    'NAF',
    'ICNN',
]
colors = [
    'red',
    'blue',
    'green'
]


import matplotlib.pyplot as plt
import numpy as np

plt.style.use('bmh')
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
plt.xlabel('Timestep')
plt.ylabel('Reward')



import os
def get_data(folders, n=100, k=100):
    X = [i * k for i in xrange(n)]
    Y = []
    for folder in folders:
        with open(os.path.join(folder, 'log.txt')) as f:
            lines = f.readlines()
        lines = map(lambda x: map(float, x.split()), lines)
        try:
            x = np.asarray(lines)[:, 0].flatten()
            y = np.asarray(lines)[:, 1].flatten()
        except:
            print folder
            continue
        z = np.ones(n) * -1e8
        z[0] = y[0]
        for u, v in zip(x[1:], y[1:]):
            idx = int((u - 1) / k) + 1
            if idx >= n:
                break
            z[idx] = v
        for i in xrange(n):
            if z[i] < -1e7:
                z[i] = z[i - 1]

        Y.append(z)

    Y = np.asarray(Y)
    Ysdom = 1.96 * np.std(Y, axis=0) / np.sqrt(Y.shape[0])
    Ymean = np.mean(Y, axis=0)

    return X, Ymean - Ysdom, Ymean + Ysdom, Ymean


if __name__ == '__main__':
    lines = []
    for name, item, color in zip(names, folders, colors):
        X, Ymin, Ymax, Ymean = get_data(item, FLAGS.total / FLAGS.train, FLAGS.train)
        line, = plt.plot(X, Ymean, label=name, color=color)
        lines.append(line)
        plt.fill_between(X, Ymin, Ymax, alpha=0.1, color=color)
    plt.legend(handles=lines, loc=2)
    plt.savefig(FLAGS.data + '/result.pdf')
