def homework(train_X, train_y, test_X):
    from sklearn.utils import shuffle
    import tensorflow as tf
    import numpy as np

    tf.reset_default_graph()

    nodes = [784,256,256,10]
    dropout = [0.2, 0.2]
    n_epochs = 30
    batch_size = 50
    eps = 0.01 * batch_size
    rng = np.random.RandomState(1234)
    random_state = 42

    x = tf.placeholder(tf.float32, [None, nodes[0]])
    t = tf.placeholder(tf.float32, [None, nodes[-1]])
    m1 = tf.placeholder(tf.float32, [nodes[1]])
    m2 = tf.placeholder(tf.float32, [nodes[2]])

    W1 = tf.Variable(rng.uniform(low=-0.08, high=0.08, size=(nodes[0], nodes[1])).astype('float32'), name='W1')
    b1 = tf.Variable(np.zeros(nodes[1]).astype('float32'), name='b1')
    W2 = tf.Variable(rng.uniform(low=-0.08, high=0.08, size=(nodes[1], nodes[2])).astype('float32'), name='W2')
    b2 = tf.Variable(np.zeros(nodes[2]).astype('float32'), name='b2')
    W3 = tf.Variable(rng.uniform(low=-0.08, high=0.08, size=(nodes[2], nodes[3])).astype('float32'), name='W3')
    b3 = tf.Variable(np.zeros(nodes[3]).astype('float32'), name='b3')
    params = [W1, b1, W2, b2, W3, b3]

    u1 = tf.matmul(x, W1) + b1
    z1 = tf.nn.sigmoid(u1) * m1
    u2 = tf.matmul(z1, W2) + b2
    z2 = tf.nn.sigmoid(u2) * m2
    u3 = tf.matmul(z2, W3) + b3
    y = tf.nn.softmax(u3)

    cost = -tf.reduce_mean(tf.reduce_sum(t*tf.log(tf.clip_by_value(y, 1e-10, 1.0)), axis=1)) # tf.log(0)によるnanを防ぐ

    gW1, gb1, gW2, gb2, gW3, gb3 = tf.gradients(cost, params)
    updates = [
        W1.assign_add(-eps*gW1),
        b1.assign_add(-eps*gb1),
        W2.assign_add(-eps*gW2),
        b2.assign_add(-eps*gb2),
        W3.assign_add(-eps*gW3),
        b3.assign_add(-eps*gb3)
    ]

    train = tf.group(*updates)

    valid = tf.argmax(y,1)

    def onehot(y):
        print(y.shape, y.shape[0])
        target_y = np.zeros((y.shape[0],10))
        target_y[np.arange(y.shape[0]), y] = 1
        return target_y
    
    def make_mask(num,p):
        return np.random.choice([0,1], size=(num,),p=[p,1-p])

    n_batchs = train_X.shape[0] // batch_size
    train_y = onehot(train_y)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(n_epochs):
            print('epoch', e)
            train_X, train_y = shuffle(train_X,train_y,random_state=random_state)
            error = 0
            for i in range(n_batchs):
                start = i * batch_size
                end = start + batch_size

                _m1 = make_mask(nodes[1], dropout[0]) 
                _m2 = make_mask(nodes[2], dropout[1]) 
                _, c = sess.run([train, cost], feed_dict={x:train_X[start:end], t: train_y[start:end], m1: _m1, m2: _m2})
                error += c
            print('error=', error)

        _m1 = np.ones(nodes[1]) * (1 - dropout[0])
        _m2 = np.ones(nodes[2]) * (1 - dropout[1])
        pred_y = sess.run(valid, feed_dict={x: test_X, m1: _m1, m2: _m2})

    return pred_y
