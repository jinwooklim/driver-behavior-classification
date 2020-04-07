import tensorflow as tf
from config import *


class ClassificationModel():
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.build_net()

    def build_net(self):
        with tf.variable_scope(self.name):
            # self.build_vgg16()
            self.build_cnn()
        # cross-entropy
        # self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y))
        # sparse-cross-entropy
        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.cost)
        # correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1)) # for cross-entropy
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), self.y) # for sparse-cross-entropy
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def build_vgg16(self, trainable=None):
        self.x = tf.placeholder(tf.float32, [None, IMG_WIDTH, IMG_HEIGHT, 3])
        # self.y = tf.placeholder(tf.float32, [None, NUM_CLASSES]) # for cross-entropy
        self.y = tf.placeholder(tf.float32, [None]) # for sparse-cross-entropy

        conv1_1 = self.conv2d(self.x, num_filters=64, name='conv1_1', trainable=trainable)
        conv1_2 = self.conv2d(conv1_1, num_filters=64, name='conv1_2', trainable=trainable)
        max_pool1 = tf.layers.max_pooling2d(conv1_2, pool_size=2, strides=2, padding='SAME')

        conv2_1 = self.conv2d(max_pool1, num_filters=128, name='conv2_1', trainable=trainable)
        conv2_2 = self.conv2d(conv2_1, num_filters=128, name='conv2_2', trainable=trainable)
        max_pool2 = tf.layers.max_pooling2d(conv2_2, pool_size=2, strides=2, padding='SAME')

        conv3_1 = self.conv2d(max_pool2, num_filters=256, name='conv3_1', trainable=trainable)
        conv3_2 = self.conv2d(conv3_1, num_filters=256, name='conv3_2', trainable=trainable)
        conv3_3 = self.conv2d(conv3_2, num_filters=256, name='conv3_3', trainable=trainable)
        max_pool3 = tf.layers.max_pooling2d(conv3_3, pool_size=2, strides=2, padding='SAME')

        conv4_1 = self.conv2d(max_pool3, num_filters=512, name='conv4_1', trainable=trainable)
        conv4_2 = self.conv2d(conv4_1, num_filters=512, name='conv4_2', trainable=trainable)
        conv4_3 = self.conv2d(conv4_2, num_filters=512, name='conv4_3', trainable=trainable)
        max_pool4 = tf.layers.max_pooling2d(conv4_3, pool_size=2, strides=2, padding='SAME')

        conv5_1 = self.conv2d(max_pool4, num_filters=512, name='conv5_1', trainable=trainable)
        conv5_2 = self.conv2d(conv5_1, num_filters=512, name='conv5_2', trainable=trainable)
        conv5_3 = self.conv2d(conv5_2, num_filters=512, name='conv5_3', trainable=trainable)
        max_pool5 = tf.layers.max_pooling2d(conv5_3, pool_size=2, strides=2, padding='SAME')

        fc6 = tf.contrib.layers.flatten(max_pool5)
        fc6 = self.fc(fc6, size=4096, name='fc6', trainable=trainable)
        fc7 = self.fc(fc6, size=4096, name='NATIN', trainable=trainable)
        self.logits = tf.layers.dense(fc7, units=NUM_CLASSES, name='fc8')

    def conv2d(self, layer, num_filters, name, ksize=3, trainable=True):
        return tf.layers.conv2d(layer, filters=num_filters, kernel_size=ksize,
                                activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5),
                                padding='SAME', trainable=trainable, use_bias=True, name=name)

    def fc(self, layer, size, name, trainable=True):
        return tf.layers.dense(layer, size, activation=tf.nn.relu,
                               trainable=trainable, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               use_bias=True, name=name)

    def build_cnn(self):
        self.x = tf.placeholder(tf.float32, [None, IMG_WIDTH, IMG_HEIGHT, 3])
        # self.y = tf.placeholder(tf.float32, [None, NUM_CLASSES]) # for cross-entropy
        self.y = tf.placeholder(tf.int64, [None]) # for sparse-cross-entropy

        layer1 = self.conv2d(self.x, num_filters=32, name='layer1', ksize=3, trainable=True)
        layer1 = tf.layers.max_pooling2d(layer1, pool_size=2, strides=2)

        layer2 = self.conv2d(layer1, num_filters=64, name='layer2', ksize=3, trainable=True)
        layer2 = tf.layers.max_pooling2d(layer2, pool_size=2, strides=2)

        layer3 = self.conv2d(layer2, num_filters=128, name='layer3', ksize=3, trainable=True)
        layer3 = tf.layers.max_pooling2d(layer3, pool_size=2, strides=2)

        dense1 = tf.contrib.layers.flatten(layer3)
        dense1 = tf.layers.dense(dense1, units=256, activation=tf.nn.relu)
        self.logits = tf.layers.dense(dense1, units=NUM_CLASSES)

    def do_prediction(self, x_test):
        return self.sess.run(self.logits, feed_dict={self.x: x_test})

    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.x: x_test, self.y: y_test})

    def train(self, x_data, y_data):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.x: x_data, self.y: y_data})