def homework(train_X, train_y, test_X):
    import numpy as np
    import tensorflow as tf
    from sklearn.utils import shuffle
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    debug = True

    print('train_X:', train_X.shape, 'train_y', train_y.shape)

    ###########################################################################
    # params
    fine_epochs = 10
    fine_batch_size = 100
    fine_opt_rate = 0.005

    rng = np.random.RandomState(1234)
    random_state = 42

    if debug:
        train_X, valid_X, train_y, valid_y = train_test_split(
            train_X,
            train_y,
            test_size=0.1,
            random_state=random_state
        )

    ###########################################################################
    # class

    class Conv:
        def __init__(self, filter_shape,
                     function=lambda x: x, strides=[1, 1, 1, 1],
                     padding='SAME'):
            # Xavier Initialization
            fan_in = np.prod(filter_shape[:3])
            fan_out = np.prod(filter_shape[:2]) * filter_shape[3]
            self.W = tf.Variable(rng.uniform(
                            low=-np.sqrt(6/(fan_in + fan_out)),
                            high=np.sqrt(6/(fan_in + fan_out)),
                            size=filter_shape
                        ).astype('float32'), name='W')
            # バイアスはフィルタごとなので, 出力フィルタ数と同じ次元数
            self.b = tf.Variable(np.zeros((filter_shape[3]), dtype='float32'),
                                 name='b')
            self.function = function
            self.strides = strides
            self.padding = padding

        def train(self, x):
            return self.f_prop(x)

        def f_prop(self, x):
            u = tf.nn.conv2d(x, self.W,
                             strides=self.strides,
                             padding=self.padding) + self.b
            return self.function(u)

    class Pooling:
        def __init__(self, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                     padding='SAME'):
            self.ksize = ksize
            self.strides = strides
            self.padding = padding

        def train(self, x):
            return self.f_prop(x)

        def f_prop(self, x):
            return tf.nn.max_pool(x, ksize=self.ksize,
                                  strides=self.strides,
                                  padding=self.padding)

    class Dropout:
        def __init__(self, p=0.5):
            self.p = p

        def train(self, x):
            return tf.nn.dropout(x, self.p)

        def f_prop(self, x):
            return tf.nn.dropout(x, 1.0)

    class Flatten:
        def train(self, x):
            return self.f_prop(x)

        def f_prop(self, x):
            return tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))

    class Dense:
        def __init__(self, in_dim, out_dim, function=lambda x: x):
            # Xavier Initialization
            self.W = tf.Variable(rng.uniform(
                            low=-np.sqrt(6/(in_dim + out_dim)),
                            high=np.sqrt(6/(in_dim + out_dim)),
                            size=(in_dim, out_dim)
                        ).astype('float32'), name='W')
            self.b = tf.Variable(np.zeros([out_dim]).astype('float32'))
            self.function = function

        def train(self, x):
            return self.f_prop(x)

        def f_prop(self, x):
            return self.function(tf.matmul(x, self.W) + self.b)

    ###########################################################################
    # layers

    # layers = [                             # (縦の次元数)x(横の次元数)x(チャネル数)
    #     Conv((5, 5, 1, 20), tf.nn.relu),   # 28x28x 1 -> 24x24x20
    #     Pooling((1, 2, 2, 1)),             # 24x24x20 -> 12x12x20
    #     Dropout(0.7),
    #     Conv((5, 5, 20, 50), tf.nn.relu),  # 12x12x20 ->  8x 8x50
    #     Pooling((1, 2, 2, 1)),             # 08x08x50 ->  4x 4x50
    #     Dropout(0.7),
    #     Flatten(),
    #     Dropout(0.6),
    #     Dense(4*4*50, 10, tf.nn.softmax)
    # ]

    layers = [                             # (縦の次元数)x(横の次元数)x(チャネル数)
        Conv((5, 5, 1, 32), tf.nn.relu),   # 28x28x 1 -> 28x28x32
        Pooling((1, 2, 2, 1)),             # 28x28x32 -> 14x14x32
        Conv((5, 5, 32, 64), tf.nn.relu),  # 14x14x32 -> 14x14x64
        Pooling((1, 2, 2, 1)),             # 14x14x64 -> 7x7x64
        Flatten(),
        Dense(7*7*64, 1024, tf.nn.relu),
        Dropout(0.5),
        Dense(1024, 10, tf.nn.softmax)
    ]

    ###########################################################################
    # fine-tuning

    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    t = tf.placeholder(tf.float32, [None, 10])

    def f_props(layers, x):
        for layer in layers:
            x = layer.f_prop(x)
        return x

    def trains(layers, x):
        for layer in layers:
            x = layer.train(x)
        return x

    y = trains(layers, x)

    cross = t * tf.log(tf.clip_by_value(y, 1e-10, 1.0))  # tf.log(0)によるnanを防ぐ
    cost = -tf.reduce_mean(tf.reduce_sum(cross, axis=1))
    train = tf.train.AdamOptimizer(fine_opt_rate).minimize(cost)

    valid = tf.argmax(f_props(layers, x), 1)

    n_batches = train_X.shape[0]//fine_batch_size

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(fine_epochs):
            train_X, train_y = shuffle(train_X, train_y,
                                       random_state=random_state)
            for i in range(n_batches):
                start = i * fine_batch_size
                end = start + fine_batch_size
                sess.run(train, feed_dict={x: train_X[start:end],
                                           t: train_y[start:end]})
            if debug:
                pred_y, valid_cost = sess.run([valid, cost],
                                              feed_dict={x: valid_X,
                                                         t: valid_y})
                f1 = f1_score(np.argmax(valid_y, 1).astype('int32'),
                              pred_y, average='macro')
                print('EPOCH:: %i, Validation cost: %.3f, Validation F1: %.3f'
                      % (epoch + 1, valid_cost, f1))

        pred_y = sess.run(valid, feed_dict={x: test_X})

    return pred_y
