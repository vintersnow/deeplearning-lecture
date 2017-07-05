def homework(train_X, train_y, test_X):
    import numpy as np
    import tensorflow as tf
    # import matplotlib.pyplot as plt
    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    # from tensorflow.examples.tutorials.mnist import input_data

    debug = False
    adam = True

    ###########################################################################
    # params
    cl = np.float(0.3)
    pretrain_batch_size = 100
    pretrain_epochs = 20
    # pretrain_eps = 0.01
    pretrain_rate = 0.01

    fine_epochs = 20
    fine_batch_size = 100
    # fine_eps = 0.01
    fine_rate = 0.01

    l2_lambda = 5e-5

    rng = np.random.RandomState(1234)
    random_state = 42

    if debug:
        train_X, valid_X, train_y, valid_y = train_test_split(
            train_X,
            train_y,
            test_size=0.1,
            random_state=random_state
        )

    class Autoencoder:
        def __init__(self, vis_dim, hid_dim, W, function=lambda x: x):
            self.W = W
            self.a = tf.Variable(np.zeros(vis_dim).astype('float32'), name='a')
            self.b = tf.Variable(np.zeros(hid_dim).astype('float32'), name='b')
            self.function = function
            self.params = [self.W, self.a, self.b]

        def encode(self, x):
            u = tf.matmul(x, self.W) + self.b
            return self.function(u)

        def decode(self, x):
            u = tf.matmul(x, tf.transpose(self.W)) + self.a
            return self.function(u)

        def f_prop(self, x):
            y = self.encode(x)
            return self.decode(y)

        def reconst_error(self, x, noise):
            tilde_x = x * noise
            reconst_x = self.f_prop(tilde_x)
            # error = -tf.reduce_mean(
            #     tf.reduce_sum(
            #         x * tf.log(tf.clip_by_value(reconst_x, 1e-10, 1.0)) +
            #         (1. - x) * tf.log(tf.clip_by_value(1. - reconst_x, 1e-10, 100.0)),
            #         axis=1
            #     )
            # )
            error = tf.reduce_mean(
                tf.reduce_sum(tf.pow(x - reconst_x, 2), 1) / 2)
            return error, reconst_x

    class Dense:
        def __init__(self, in_dim, out_dim, function,
                     dropout_p=1.0, beta1=0.1, beta2=0.1):
            self.W = tf.Variable(rng.uniform(
                low=-0.08, high=0.08, size=(in_dim, out_dim)
            ).astype('float32'), name='W')
            self.b = tf.Variable(np.zeros([out_dim]).astype('float32'))
            self.function = function
            self.params = [self.W, self.b]
            self.p = dropout_p
            self.out_dim = out_dim
            self.mk = tf.Variable(self.mask(out_dim),
                                  name='mask')

            self.mW = tf.Variable(np.zeros((in_dim, out_dim), dtype='float32'))
            self.vW = tf.Variable(np.zeros((in_dim, out_dim), dtype='float32'))
            self.mb = tf.Variable(np.zeros(out_dim, dtype='float32'))
            self.vb = tf.Variable(np.zeros(out_dim, dtype='float32'))
            self.moments = [(self.mW, self.vW),
                            (self.mb, self.vb)]
            self.moments_init = [
                self.mW.initializer,
                self.vW.initializer,
                self.mb.initializer,
                self.vb.initializer,
            ]

            self.ae = Autoencoder(in_dim, out_dim, self.W, self.function)

        def mask(self, n, dropout=False):
            if dropout:
                return np.random.choice([0, 1], size=(n,),
                                        p=[1-self.p, self.p]
                                        ).astype('float32')
            else:
                return np.ones(n, dtype=np.float32) * self.p

        def f_prop(self, x, dropout=False):
            self.mk.assign(self.mask(self.out_dim, dropout))
            u = tf.matmul(x, self.W) + self.b
            self.z = self.function(u) * self.mk
            return self.z

        def pretrain(self, x, noise):
            cost, reconst_x = self.ae.reconst_error(x, noise)
            return cost, reconst_x

        def reset_moments(self, sess):
            sess.run(self.moments_init)

    ###########################################################################

    def sgd(cost, params, rate=np.float32(0.5)):
        g_params = tf.gradients(cost, params)
        updates = []
        for param, g_param in zip(params, g_params):
            if g_param is not None:
                updates.append(param.assign_add(-rate*g_param))
        return updates

    def adam(cost, params, moments, time,
             alpha=np.float32(0.001), eps=np.float32(1e-8),
             beta1=np.float32(0.9), beta2=np.float32(0.99)):
        g_params = tf.gradients(cost, params)

        updates = []
        for param, g_param, moment in zip(params, g_params, moments):
            if g_param is None:
                continue
            m = moment[0]
            v = moment[1]
            updates.append(m.assign(beta1 * m + (1 - beta1) * g_param))
            updates.append(v.assign(beta2 * v + (1 - beta2) * g_param ** 2))
            m_hat = m / (1 - beta1 ** time)
            v_hat = v / (1 - beta2 ** time)
            updates.append(param.assign_add(
                -alpha * m_hat / (tf.sqrt(v_hat) + eps)
            ))
        return updates

    ###########################################################################
    # Layers

    nodes = [784, 512, 256, 128, 10]
    layers = [
        Dense(nodes[0], nodes[1], tf.nn.relu, 0.5),
        Dense(nodes[1], nodes[2], tf.nn.relu, 0.5),
        Dense(nodes[2], nodes[3], tf.nn.relu, 0.5),
        Dense(nodes[3], nodes[4], tf.nn.softmax)
    ]

    ###########################################################################
    # pre-training
    X = np.copy(train_X)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for l, layer in enumerate(layers[:-1]):
        n_batches = X.shape[0] // pretrain_batch_size

        x = tf.placeholder(tf.float32)
        noise = tf.placeholder(tf.float32)
        time = tf.placeholder(tf.float32)

        cost, reconst_x = layer.pretrain(x, noise)
        params = layer.params
        moments = layer.moments
        if adam:
            train = adam(cost, params, moments, time)
        else:
            train = sgd(cost, params, pretrain_rate)

        encode = layer.f_prop(x)

        for epoch in range(1, pretrain_epochs+1):
            X = shuffle(X, random_state=random_state)
            err_all = []
            for i in range(n_batches):
                start = i * pretrain_batch_size
                end = start + pretrain_batch_size

                _noise = rng.binomial(size=X[start:end].shape, n=1, p=1-cl)
                _, err = sess.run([train, cost],
                                  feed_dict={x: X[start:end], noise: _noise,
                                             time: epoch})
                err_all.append(err)
            if debug:
                print('Pretraining:: layer: %d, Epoch: %d, Error: %lf'
                      % (l+1, epoch, np.mean(err)))
        X = sess.run(encode, feed_dict={x: X})

    ###########################################################################
    # fine-tuning
    x = tf.placeholder(tf.float32, [None, 784])
    t = tf.placeholder(tf.float32, [None, 10])
    time = tf.placeholder(tf.float32)

    def f_props(layers, x, dropout=False):
        params = []
        moments = []
        for layer in layers:
            layer.reset_moments(sess)
            x = layer.f_prop(x, dropout)
            params += layer.params
            moments += layer.moments
        return x, params, moments

    y, params, moments = f_props(layers, x, True)
    cost = -tf.reduce_mean(tf.reduce_sum(
        t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), 1))
    for l in layers:
        cost += l2_lambda * tf.nn.l2_loss(l.W)

    if adam:
        updates = adam(cost, params, moments, time)
    else:
        updates = sgd(cost, params, fine_rate)

    train = tf.group(*updates)

    vy, params, moments = f_props(layers, x, False)
    valid = tf.argmax(vy, 1)

    n_batches = train_X.shape[0] // fine_batch_size

    for epoch in range(1, fine_epochs+1):
        train_X, train_y = shuffle(train_X, train_y, random_state=random_state)
        for i in range(n_batches):
            start = i * fine_batch_size
            end = start + fine_batch_size
            sess.run(train,
                     feed_dict={x: train_X[start:end], t: train_y[start:end],
                                time: epoch})
        if debug:
            pred_y, valid_cost = sess.run([valid, cost],
                                          feed_dict={x: valid_X, t: valid_y})
            f1 = f1_score(np.argmax(valid_y, 1).astype('int32'),
                          pred_y, average='macro')
            print('EPOCH: %i, Validation cost: %.3f Validation F1: %.3f'
                  % (epoch, valid_cost, f1))

    pred_y = sess.run(valid, feed_dict={x: test_X})
    print(pred_y.shape)
    sess.close()
    return pred_y
