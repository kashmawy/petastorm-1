#  Copyright (c) 2017-2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###
# Adapted to petastorm dataset using original contents from
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py
###

from __future__ import division, print_function

import argparse
import tensorflow as tf

from examples.mnist import DEFAULT_MNIST_DATA_PATH
from petastorm.reader import Reader


def train_and_test(train_reader, test_reader, epochs, batch_size, log_interval):
    """
    Train a model for epochs with a batch size batch_size, printing accuracy every log_interval.
    :param train_reader: A reader for train data.
    :param test_reader: A reader for test data.
    :param epochs: The epochs to train for.
    :param batch_size: The batch size for training.
    :param log_interval: The interval used to print the accuracy.
    :return:
    """
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.int64, [None])

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.losses.sparse_softmax_cross_entropy on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Train
    print("Training model for {0} epochs with batch size {1} and log interval {2}".format(
        epochs, batch_size, log_interval
    ))
    for i in range(epochs):
        batch_x, batch_y = get_batch(train_reader, batch_size)
        sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})

        if (i % log_interval) == 0:
            batch_x, batch_y = get_batch(test_reader, batch_size)
            correct_prediction = tf.equal(tf.argmax(y, 1), y_)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print("Traing for {0} epochs, accuracy of the model: {1:.2f}".format(
                i,
                sess.run(accuracy, feed_dict={x: batch_x, y_: batch_y})))


def get_batch(reader, batch_size):
    """
    Gets a batch of size batch_size from the reader.
    Transforms the image to be of shape 784.
    Returns a tuple, first item is the images batched and the second item is the digit batched.
    :param reader: The reader to batch the data from.
    :param batch_size: The batch size.
    :return: A tuple, where the first item is the images batched, and the second item is the digits batched.
    """
    x = []
    y = []
    for _ in range(batch_size):
        item = next(reader)
        x.append(item.image.reshape(784))
        y.append(item.digit)
    return x, y


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Petastorm Tensorflow MNIST Example')
    default_dataset_url = 'file://{}'.format(DEFAULT_MNIST_DATA_PATH)
    parser.add_argument('--dataset-url', type=str,
                        default=default_dataset_url, metavar='S',
                        help='hdfs:// or file:/// URL to the MNIST petastorm dataset'
                             '(default: %s)' % default_dataset_url)
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status (default: 10)')
    args = parser.parse_args()

    train_reader = Reader('{}/train'.format(args.dataset_url))
    test_reader = Reader('{}/test'.format(args.dataset_url))
    train_and_test(
        train_reader=train_reader,
        test_reader=test_reader,
        epochs=args.epochs,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
    )


if __name__ == '__main__':
    main()
