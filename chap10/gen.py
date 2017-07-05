def generate(test_X, test_y):
    global e_vocab_size, j_vocab_size, sess, x, d, decoder_pre, decoder_post, e_i2w, j_i2w, h_enc, c_enc
    print('generate', e_vocab_size, j_vocab_size, sess, x, d)

    t_0 = tf.constant(0)
    y_0 = tf.placeholder(tf.int32, [None, None], name='y_0')
    h_0 = tf.placeholder(tf.float32, [None, None], name='h_0')
    c_0 = tf.placeholder(tf.float32, [None, None], name='c_0')
    f_0 = tf.cast(tf.zeros_like(y_0[:, 0]), dtype=tf.bool) # バッチ内の各サンプルに対して</s>が出たかどうかのflag
    f_0_size = tf.reduce_sum(tf.ones_like(f_0, dtype=tf.int32))
    max_len = tf.placeholder(tf.int32, name='max_len') # iterationの繰り返し回数の限度

    def f_props_test(layers, x_t):
        for layer in layers:
            x_t = layer.f_prop_test(x_t)
        return x_t

    def cond(t, h_t, c_t, y_t, f_t):
        num_true = tf.reduce_sum(tf.cast(f_t, tf.int32)) # Trueの数
        unfinished = tf.not_equal(num_true, f_0_size)
        return tf.logical_and(t + 1 < max_len, unfinished)

    def body(t, h_tm1, c_tm1, y, f_tm1):
        y_tm1 = y[:, -1]

        decoder_pre[1].h_0 = h_tm1
        decoder_pre[1].c_0 = c_tm1
        h_t, c_t = f_props_test(decoder_pre, y_tm1)
        y_t = tf.cast(tf.argmax(f_props_test(decoder_post, h_t), axis=1), tf.int32)

        y = tf.concat([y, y_t[:, np.newaxis]], axis=1)

        f_t = tf.logical_or(f_tm1, tf.equal(y_t, 1)) # flagの更新

        return [t + 1, h_t, c_t, y, f_t]

    print('hogehoge------------------------------')
    res = tf.while_loop(
        cond,
        body,
        loop_vars=[t_0, h_0, c_0, y_0, f_0],
        shape_invariants=[
            t_0.get_shape(),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None])
        ]
    )

    test_X_mb = pad_sequences(test_X, padding='post', value=-1)
    _y_0 = np.zeros_like(test_X, dtype='int32')[:, np.newaxis]
    _h_enc, _c_enc = sess.run([h_enc, c_enc], feed_dict={x: test_X_mb})
    _h_0 = _h_enc[:, -1, :]
    _c_0 = _c_enc[:, -1, :]

    print('ffffffff------------------------------')

    _, _, _, pred_y, _ = sess.run(res, feed_dict={
        y_0: _y_0,
        h_0: _h_0,
        c_0: _c_0,
        max_len: 100
    })

    for i in range(10):
        origy = test_X[i][1:-1]
        predy = list(pred_y[i])
        truey = test_y[i][1:-1]

        print('origin:', ' '.join([e_i2w[com] for com in origy]))
        print('output:', ' '.join([j_i2w[com] for com in predy[1:]]))
        print('correct:', ' '.join([j_i2w[com] for com in truey]))

