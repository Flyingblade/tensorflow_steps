"""
    A DNN demo with regulation(l2).
    Trying TensorFlow "collection"
    Create by: FlyingBlade
    Create Time: 2018/6/7 20:26
"""
import tensorflow as tf
import numpy as np
import os
import math
from DataGenerator import DataGenerator

# config
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
sess = tf.Session(config=config)


def get_weight(shape, l2):
    '''
    Define a Variable with shape, and add its l2 loss into collection.
    :param shape: The shape of variable
    :param l2: The l2
    :return: Variable
    '''
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection("loss", tf.contrib.layers.l2_regularizer(l2)(var))
    return var


# h-params
lr = 0.001
l2 = 0.01
batch_size = 8
epochs = 1000

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
dnn_layers = [2, 8, 16, 32, 4, 1]
y = x
for i in range(1, len(dnn_layers)):
    in_dim = dnn_layers[i - 1]
    out_dim = dnn_layers[i]
    w_i = get_weight((in_dim, out_dim), l2)
    bias_i = tf.Variable(tf.random_uniform([out_dim], minval=-1, maxval=1, dtype=tf.float32))
    y = tf.nn.relu(tf.matmul(y, w_i) + bias_i)

mse_loss = tf.reduce_mean(tf.square(y - y_))
tf.add_to_collection("loss", mse_loss)

# summarize l2 loss and mse loss
loss = tf.add_n(tf.get_collection("loss"))
train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

dataset_size = 128
sess.run(tf.global_variables_initializer())
df_x, df_y = DataGenerator(dataset_size, seed=1, func=lambda x: math.log(x[0]) + x[1]**2 - math.exp(x[1])).generate()
for epoch in range(epochs):
    for i in range(int(math.ceil(dataset_size / batch_size))):
        start = batch_size * i
        end = min(dataset_size, start + batch_size)
        sess.run(train_step, feed_dict={x: df_x[start:end], y_: df_y[start:end]})
    mse = sess.run(mse_loss, {x: df_x, y_: df_y})
    epoch_loss = sess.run(loss, feed_dict={x: df_x, y_: df_y})
    print(epoch, 'mse:', mse, 'mse+l2:', epoch_loss)
    random_state = np.random.get_state()
    np.random.shuffle(df_x)
    np.random.set_state(random_state)
    np.random.shuffle(df_y)
