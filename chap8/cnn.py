def homework(train_X, train_y, test_X):
    import numpy as np
    import tensorflow as tf

    from sklearn.utils import shuffle
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    # from keras.datasets import cifar10

    ZCA_FLAG = True
    VALID = True

    learning_rate = 0.001

    if VALID:
        train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y,
                                                              test_size=0.1,
                                                              random_state=42)

        print('train_X:', train_X.shape, 'train_y', train_y.shape)
        print('valid_X', valid_X.shape, 'valid_y', valid_y.shape)
    else:
        print('train_X:', train_X.shape, 'train_y', train_y.shape)

    rng = np.random.RandomState(1234)
    random_state = 42

    class ZCAWhitening:
        def __init__(self, epsilon=1e-4):
            self.epsilon = epsilon
            self.mean = None
            self.ZCA_matrix = None

        def fit(self, x):
            x = x.reshape(x.shape[0], -1)
            self.mean = np.mean(x, axis=0)
            x -= self.mean
            cov_matrix = np.dot(x.T, x) / x.shape[0]
            A, d, _ = np.linalg.svd(cov_matrix)
            self.ZCA_matrix = np.dot(np.dot(A, np.diag(1. / np.sqrt(d + self.epsilon))), A.T)

        def transform(self, x):
            shape = x.shape
            x = x.reshape(x.shape[0], -1)
            x -= self.mean
            x = np.dot(x, self.ZCA_matrix.T)
            return x.reshape(shape)

    class BatchNorm:
        def __init__(self, out_dim, epsilon=np.float32(1e-5)):
            self.gamma = tf.Variable(np.ones([out_dim], dtype='float32'), name='gamma')
            self.beta = tf.Variable(np.zeros([out_dim], dtype='float32'), name='beta')
            self.epsilon = epsilon

        def f_prop(self, x):
            mean, var = tf.nn.moments(x, axes=[0, 1, 2])
            return tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, 0.001,
                scale_after_normalization=True)

    class Conv:
        def __init__(self, filter_shape, function=lambda x: x, strides=[1, 1, 1, 1], padding='SAME'):
            # Xavier
            fan_in = np.prod(filter_shape[:3])
            fan_out = np.prod(filter_shape[:2]) * filter_shape[3]
            self.W = tf.Variable(rng.uniform(
                            low=-np.sqrt(6/(fan_in + fan_out)),
                            high=np.sqrt(6/(fan_in + fan_out)),
                            size=filter_shape
                        ).astype('float32'), name='W')
            self.b = tf.Variable(np.zeros((filter_shape[3]), dtype='float32'), name='b')  # バイアスはフィルタごと
            self.function = function
            self.strides = strides
            self.padding = padding

        def f_prop(self, x):
            u = tf.nn.conv2d(x, self.W, strides=self.strides, padding=self.padding) + self.b
            return self.function(u)

    class Pooling:
        def __init__(self, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'):
            self.ksize = ksize
            self.strides = strides
            self.padding = padding

        def f_prop(self, x):
            return tf.nn.max_pool(x, ksize=self.ksize, strides=self.strides, padding=self.padding)

    class GlobalPool:
        def __init__(self, axis):
            self.axis = axis

        def f_prop(self, x):
            res = tf.reduce_mean(x, self.axis)
            return res

    class Flatten:
        def f_prop(self, x):
            return tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))

    class Dense:
        def __init__(self, in_dim, out_dim, function=lambda x: x):
            # Xavier
            self.W = tf.Variable(rng.uniform(
                            low=-np.sqrt(6/(in_dim + out_dim)),
                            high=np.sqrt(6/(in_dim + out_dim)),
                            size=(in_dim, out_dim)
                        ).astype('float32'), name='W')
            self.b = tf.Variable(np.zeros([out_dim]).astype('float32'))
            self.function = function

        def f_prop(self, x):
            return self.function(tf.matmul(x, self.W) + self.b)

    class Activation:
        def __init__(self, function=lambda x: x):
            self.function = function

        def f_prop(self, x):
            return self.function(x)

    class ConvLayer:
        def __init__(self, filter_shape, activation=tf.nn.relu, function=lambda x: x, strides=[1, 1, 1, 1], padding='SAME'):
            self.conv = Conv(filter_shape, function, strides)
            self.bach_norm = BatchNorm(filter_shape[3])
            self.activation = Activation(activation)

        def f_prop(self, x):
            c = self.conv.f_prop(x)
            b = self.bach_norm.f_prop(c)
            return self.activation.f_prop(b)

    class ResidualBlock:
        def __init__(self, input_dim, output_dim, down_sample, projection=False):
            self.down_sample = down_sample
            self.projection = projection
            self.conv1 = ConvLayer((3, 3, input_dim, output_dim))
            self.conv2 = ConvLayer((3, 3, output_dim, output_dim))
            self.inout_diff = output_dim - input_dim
            if projection:
                self.proj_conv = Conv((1, 1, input_dim, output_dim), strides=[1, 2, 2, 1])

        def f_prop(self, x):
            if self.down_sample:
                _filter = [1, 2, 2, 1]
                x = tf.nn.max_pool(x, ksize=_filter, strides=_filter, padding='SAME')

            if self.inout_diff != 0:
                if self.projection:
                    input_layer = self.proj_conv.f_prop(x)
                else:
                    input_layer = tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, self.inout_diff]])
            else:
                input_layer = x

            c1 = self.conv1.f_prop(x)
            c2 = self.conv2.f_prop(c1)
            res = c2 + input_layer

            return res

    def resnet(n):
        if n < 20 or (n - 20) % 12 != 0:
            print('ResNet depth invalid')
            raise

        num_conv = int((n - 20) / 12 + 1)
        layers = []

        with tf.variable_scope('conv1'):
                conv1 = ConvLayer([3, 3, 3, 16])
                layers.append(conv1)

        for i in range(num_conv):
            with tf.variable_scope('conv2_%d' % (i+1)):
                conv2_x = ResidualBlock(16, 16, False)
                conv2 = ResidualBlock(16, 16, False)
                layers.append(conv2_x)
                layers.append(conv2)

        for i in range(num_conv):
            down_sample = True if i == 0 else False
            input_dim = 16 if down_sample else 32
            with tf.variable_scope('conv3_%d' % (i+1)):
                conv3_x = ResidualBlock(input_dim, 32, down_sample)
                conv3 = ResidualBlock(32, 32, False)
                layers.append(conv3_x)
                layers.append(conv3)

        for i in range(num_conv):
            down_sample = True if i == 0 else False
            input_dim = 32 if down_sample else 64
            with tf.variable_scope('conv4_%d' % (i+1)):
                conv4_x = ResidualBlock(input_dim, 64, down_sample)
                conv4 = ResidualBlock(64, 64, False)
                layers.append(conv4_x)
                layers.append(conv4)

        with tf.variable_scope('fc'):
            layers.append(GlobalPool([1, 2]))
            # assert global_pool.get_shape().as_list()[1:] == [64]

            layers.append(Dense(64, 10, tf.nn.softmax))

        return layers

    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    t = tf.placeholder(tf.float32, [None, 10])

    def f_props(layers, x):
        for layer in layers:
            x = layer.f_prop(x)
        return x

    layers = resnet(20)
    # layers = resnet(56)

    y = f_props(layers, x)

    cost = -tf.reduce_mean(tf.reduce_sum(t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), axis=1))
    train = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cost)

    valid = tf.argmax(y, 1)

    def gcn(x):
        mean = np.mean(x, axis=(1, 2, 3), keepdims=True)
        std = np.std(x, axis=(1, 2, 3), keepdims=True)
        return (x - mean)/std

    if ZCA_FLAG:
        print('ZCA')
        zca = ZCAWhitening()
        zca.fit(gcn(train_X))
        zca_train_X = zca.transform(gcn(train_X))
        zca_train_y = train_y[:]
        if VALID:
            zca_valid_X = zca.transform(gcn(valid_X))
            zca_valid_y = valid_y[:]

    zca_train_X = train_X[:]
    zca_train_y = train_y[:]

    n_epochs = 100
    batch_size = 100
    n_batches = train_X.shape[0]//batch_size

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    print('start train ')
    for epoch in range(n_epochs):
        zca_train_X, zca_train_y = shuffle(zca_train_X, zca_train_y, random_state=random_state)
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            sess.run(train, feed_dict={x: zca_train_X[start:end], t: zca_train_y[start:end]})
        if VALID:
            pred_y, valid_cost = sess.run([valid, cost], feed_dict={x: zca_valid_X, t: zca_valid_y})
            print('EPOCH:: %i, Validation cost: %.3f, Validation F1: %.3f' % (epoch + 1, valid_cost, f1_score(np.argmax(valid_y, 1).astype('int32'), pred_y, average='macro')))
        else:
            print('epoch ', epoch)

    zca_test_X = zca.transform(gcn(test_X)) if ZCA_FLAG else test_X
    pred_y = sess.run(valid, feed_dict={x: zca_test_X})

    return pred_y
