def homework(train_X, train_y, test_X):
    import numpy as np
    import tensorflow as tf
    import time

    from sklearn.utils import shuffle
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    BN_EPSILON = 0.001
    learning_rate = 0.1
    num_residual_blocks = 5
    random_state = 42
    rng = np.random.RandomState(1234)

    ZCA_FLAG = True
    VALID = False

    ########################################################################################################
    # make data
    flip_train_X = train_X[:, :, ::-1, :]

    padded = np.pad(train_X, ((0, 0), (4, 4), (4, 4), (0, 0)), mode='constant')
    crops = rng.randint(8, size=(len(train_X), 2))
    cropped_train_X = [padded[i, c[0]:(c[0]+32), c[1]:(c[1]+32), :] for i, c in enumerate(crops)]
    cropped_train_X = np.array(cropped_train_X)

    train_X = np.concatenate((train_X, flip_train_X, cropped_train_X), axis=0)
    train_y = np.concatenate((train_y, train_y, train_y), axis=0)

    if VALID:
        train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y,
                                                              test_size=0.1,
                                                              random_state=42)

        print('train_X:', train_X.shape, 'train_y', train_y.shape)
        print('valid_X', valid_X.shape, 'valid_y', valid_y.shape)
    else:
        print('train_X:', train_X.shape, 'train_y', train_y.shape)

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

    def create_variables(name, shape, initializer=tf.uniform_unit_scaling_initializer(factor=1.0)):
        '''
        :param name: A string. The name of the new variable
        :param shape: A list of dimensions
        :param initializer: Use
        layers.
        :return: The created variable
        '''

        new_variables = tf.Variable(initializer(shape=shape))
        return new_variables

    def output_layer(input_layer, num_labels):
        '''
        :param input_layer: 2D tensor
        :param num_labels: int. How many output labels in total? (10 for cifar10 and 100 for cifar100)
        :return: output layer Y = WX + B
        '''
        input_dim = input_layer.get_shape().as_list()[-1]
        fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels],
                                initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())

        fc_h = tf.matmul(input_layer, fc_w) + fc_b
        return fc_h

    def batch_normalization_layer(input_layer, dimension):
        '''
        Helper function to do batch normalziation
        :param input_layer: 4D tensor
        :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
        :return: the 4D tensor after being normalized
        '''

        mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])

        gamma = tf.Variable(np.ones(dimension, dtype='float32'), name='gamma')
        beta = tf.Variable(np.zeros(dimension, dtype='float32'), name='beta')

        bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

        return bn_layer

    def conv_bn_relu_layer(input_layer, filter_shape, stride):
        '''
        A helper function to conv, batch normalize and relu the input tensor sequentially
        :param input_layer: 4D tensor
        :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
        :param stride: stride size for conv
        :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
        '''

        out_channel = filter_shape[-1]
        filter = create_variables(name='conv', shape=filter_shape)

        conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
        bn_layer = batch_normalization_layer(conv_layer, out_channel)

        output = tf.nn.relu(bn_layer)
        return output

    def bn_relu_conv_layer(input_layer, filter_shape, stride):
        '''
        A helper function to batch normalize, relu and conv the input layer sequentially
        :param input_layer: 4D tensor
        :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
        :param stride: stride size for conv
        :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
        '''

        in_channel = input_layer.get_shape().as_list()[-1]

        bn_layer = batch_normalization_layer(input_layer, in_channel)
        relu_layer = tf.nn.relu(bn_layer)

        filter = create_variables(name='conv', shape=filter_shape)
        conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
        return conv_layer

    def residual_block(input_layer, output_channel, first_block=False):
        '''
        Defines a residual block in ResNet
        :param input_layer: 4D tensor
        :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
        :param first_block: if this is the first residual block of the whole network
        :return: 4D tensor.
        '''
        input_channel = input_layer.get_shape().as_list()[-1]

        # When it's time to "shrink" the image size, we use stride = 2
        if input_channel * 2 == output_channel:
            increase_dim = True
            stride = 2
        elif input_channel == output_channel:
            increase_dim = False
            stride = 1
        else:
            raise ValueError('Output and input channel does not match in residual blocks!!!')

        # The first conv layer of the first residual block does not need to be normalized and relu-ed.
        with tf.variable_scope('conv1_in_block'):
            if first_block:
                filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
                conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
            else:
                conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride)

        with tf.variable_scope('conv2_in_block'):
            conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1)

        # When the channels of input layer and conv2 does not match, we add zero pads to increase the
        #  depth of input layers
        if increase_dim is True:
            pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='VALID')
            padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                          input_channel // 2]])
        else:
            padded_input = input_layer

        output = conv2 + padded_input
        return output

    def inference(input_tensor_batch, n, reuse):
        '''
        The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
        :param input_tensor_batch: 4D tensor
        :param n: num_residual_blocks
        :param reuse: To build train graph, reuse=False. To build validation graph and share weights
        with train graph, resue=True
        :return: last layer in the network. Not softmax-ed
        '''

        layers = []
        with tf.variable_scope('conv0', reuse=reuse):
            conv0 = conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, 16], 1)
            layers.append(conv0)

        for i in range(n):
            with tf.variable_scope('conv1_%d' % i, reuse=reuse):
                if i == 0:
                    conv1 = residual_block(layers[-1], 16, first_block=True)
                else:
                    conv1 = residual_block(layers[-1], 16)
                layers.append(conv1)

        for i in range(n):
            with tf.variable_scope('conv2_%d' % i, reuse=reuse):
                conv2 = residual_block(layers[-1], 32)
                layers.append(conv2)

        for i in range(n):
            with tf.variable_scope('conv3_%d' % i, reuse=reuse):
                conv3 = residual_block(layers[-1], 64)
                layers.append(conv3)
            assert conv3.get_shape().as_list()[1:] == [8, 8, 64]

        with tf.variable_scope('fc', reuse=reuse):
            in_channel = layers[-1].get_shape().as_list()[-1]
            bn_layer = batch_normalization_layer(layers[-1], in_channel)
            relu_layer = tf.nn.relu(bn_layer)
            global_pool = tf.reduce_mean(relu_layer, [1, 2])

            assert global_pool.get_shape().as_list()[-1:] == [64]
            output = output_layer(global_pool, 10)
            layers.append(output)

        return layers[-1]

    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    t = tf.placeholder(tf.float32, [None, 10])

    logits = inference(x, num_residual_blocks, reuse=False)

    y = tf.nn.softmax(logits)
    loss = -tf.reduce_mean(tf.reduce_sum(t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), axis=1))
    # full_loss = tf.add_n([loss] + regu_losses)
    full_loss = loss

    train = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(full_loss)

    valid = tf.argmax(y, 1)

    n_epochs = 10
    batch_size = 100
    n_batches = train_X.shape[0]//batch_size

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

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
    else:
        zca_train_X = train_X[:]
        zca_train_y = train_y[:]
        if VALID:
            zca_valid_X = valid_X[:]
            zca_valid_y = valid_y[:]

    print('start train ')
    for epoch in range(n_epochs):
        start_time = time.time()
        zca_train_X, zca_train_y = shuffle(zca_train_X, zca_train_y, random_state=random_state)
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            sess.run([train], feed_dict={x: zca_train_X[start:end], t: zca_train_y[start:end]})

        duration = time.time() - start_time

        if VALID:
            pred_y, valid_cost = sess.run([valid, loss], feed_dict={x: zca_valid_X, t: zca_valid_y})
            f1 = f1_score(np.argmax(valid_y, 1).astype('int32'), pred_y, average='macro')
            print('EPOCH:: %i, TIME: %.3f, Validation cost: %.3f, Validation F1: %.3f' % (epoch + 1, duration, valid_cost, f1))
        else:
            print('EPOCH: %i, TIME: %.3f' % (epoch, duration))

    test_batch_size = batch_size
    pred_y = np.zeros(test_X.shape[0])
    zca_test_X = zca.transform(gcn(test_X)) if ZCA_FLAG else test_X

    print('test_X: ', test_X.shape)
    for i in range(test_X.shape[0] // test_batch_size):
        start = i * test_batch_size
        end = start + test_batch_size
        pred_y[start:end] = sess.run(valid, feed_dict={x: zca_test_X[start:end]})

    print(pred_y)

    sess.close()

    return pred_y
