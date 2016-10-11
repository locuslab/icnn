# Scale weights.
# tflearn.is_training(True)
# I = npr.randint(nTrain, size=10000)
# xBatch = trainX[I, :]
# yBatch = trainY[I, :]
# W_z_zs = [x for x in self.theta_ if 'zu_proj/W' in x.name]
# W_z_ys = [x for x in self.theta_ if '_yu/' in x.name]
# W_z_us = [x for x in self.theta_ if re.match('z[0-9]*_u/W', x.name) is not None]
# b_z = [x for x in self.theta_ if re.match('z[0-9]*_u/b', x.name) is not None]
# z_zs, z_ys, z_us = self.sess.run([self.z_zs, self.z_ys, self.z_us],
#                                  feed_dict={self.x_: xBatch, self.y_: yBatch})
# print("=== Original ===")
# new_bzs = []
# scale = 0.1
# for i in range(self.nLayers+1):
#     sz = self.szs[i] if i < self.nLayers else 1
#     b_zi_new = np.zeros(sz)

#     z_ys_i_mean = np.mean(z_ys[i], axis=0)
#     z_ys_i_std = np.std(z_ys[i], axis=0)
#     print('y.std:', z_ys_i_std[:5])
#     # self.sess.run(tf.assign(W_z_ys[i],
#     #                         tf.matmul(W_z_ys[i], tf.diag(scale/z_ys_i_std))))
#     # b_zi_new -= z_ys_i_mean/z_ys_i_std
#     # b_zi_new -= z_ys_i_mean

#     z_us_i_mean = np.mean(z_us[i], axis=0)
#     z_us_i_std = np.std(z_us[i], axis=0)
#     print('u.std:', z_us_i_std[:5])
#     # self.sess.run(tf.assign(W_z_us[i],
#     #                         tf.matmul(W_z_us[i], tf.diag(scale/z_us_i_std))))
#     # b_zi_new -= z_us_i_mean/z_us_i_std
#     # b_zi_new -= z_us_i_mean

#     new_bzs.append(b_zi_new)
#     if i == 0:
#         self.sess.run(tf.assign(b_z[i], new_bzs[i]))

# for i in range(self.nLayers):
#     z_zs, = self.sess.run([self.z_zs],
#                          feed_dict={self.x_: xBatch, self.y_: yBatch})
#     z_zs_i_mean = np.mean(z_zs[i], axis=0)
#     z_zs_i_std = np.std(z_zs[i], axis=0)
#     print('z.std:', z_zs_i_std[:5])
#     self.sess.run(tf.assign(W_z_zs[i],
#                             tf.matmul(W_z_zs[i], tf.diag(scale/z_zs_i_std))))
#     new_bzs[i+1] -= z_zs_i_mean/z_zs_i_std
#     self.sess.run(tf.assign(b_z[i+1], new_bzs[i+1]))

# print("=== Scaled ===")
# z_zs, z_ys, z_us = self.sess.run([self.z_zs, self.z_ys, self.z_us],
#                                  feed_dict={self.x_: xBatch, self.y_: yBatch})
# for i in range(self.nLayers+1):
#     if i > 0:
#         z_zs_i_std = np.std(z_zs[i-1], axis=0)
#         print('z.std:', z_zs_i_std[:5])

#     z_ys_i_std = np.std(z_ys[i], axis=0)
#     print('y.std:', z_ys_i_std[:5])

#     z_us_i_std = np.std(z_us[i], axis=0)
#     print('u.std:', z_us_i_std[:5])

# import IPython; IPython.embed(); sys.exit(-1)
# sys.exit(-1)

def f_shallow(self, x, y, szs, reuse=False):
    assert(len(szs) == 2)
    act = tf.nn.relu
    # act = tf.nn.softplus
    xy = tf.concat(1, (x, y))

    if reuse:
        tf.get_variable_scope().reuse_variables()

    # with tf.variable_scope('u0') as s:
    #     W_x = getW(self.nFeatures, szs[0], 'x')
    #     b = getB(szs[0])
    #     u0 = tf.identity(tf.matmul(x, W_x) + b, name='preact')
    #     tf.histogram_summary(u0.name, u0)
    #     u0 = act(u0, name='act')
    #     tf.histogram_summary(u0.name, u0)
    #     u0 = tflearn.layers.normalization.batch_normalization(u0, reuse=reuse,
    #                                                         scope=s, name='bn')
    #     tf.histogram_summary(u0.name, u0)

    # ui = u0
    # i,j = 0,1
    # with tf.variable_scope('u'+str(j)) as s:
    #     W_u = getW(szs[i], szs[j], 'u')
    #     b = getB(szs[j])
    #     uj = tf.identity(tf.matmul(ui, W_u) + b, name='preact')
    #     tf.histogram_summary(uj.name, uj)
    #     if len(szs) > i+2:
    #         uj = act(uj, name='act')
    #         tf.histogram_summary(uj.name, uj)
    #         uj = tflearn.layers.normalization.batch_normalization(uj, reuse=reuse,
    #                                                               scope=s, name='bn')
    #         tf.histogram_summary(uj.name, uj)

    # zSz = 10
    # with tf.variable_scope('z0'):
    #     z0 = act(tf.matmul(y, getW(self.nLabels, zSz, 'y')) + getB(zSz))
    # with tf.variable_scope('z1'):
    #     z1 = tf.reshape(tf.matmul(z0, getW(zSz, 1, 'z')) + getB(1), [-1])

    # zk = -tf.reduce_sum(tf.mul(x, y), 1)

    return tf.reshape(zk, [-1], name='energies')
