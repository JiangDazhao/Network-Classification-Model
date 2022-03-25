import time
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import metrics

from tensorflow.examples.tutorials.mnist import input_data
# start tensorflow interactiveSession
import tensorflow as tf

# Note: if class numer is 2 or 20, please edit the variable named "num_classes" in /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py"
DATA_DIR = sys.argv[1]
CLASS_NUM = int(sys.argv[2])
TRAIN_ROUND = int(sys.argv[3])

folder = os.path.split(DATA_DIR)[1]

sess = tf.InteractiveSession()

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', DATA_DIR, 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)


# function: find a element in a list
def find_element_in_list(element, list_element):
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return -1
# weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Create the model
# placeholder
x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, CLASS_NUM])

# first convolutinal layer
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
w_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
w_fc2 = weight_variable([1024, CLASS_NUM])
b_fc2 = bias_variable([CLASS_NUM])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# define var&op of training&testing
actual_label = tf.argmax(y_, 1)
label, idx, count = tf.unique_with_counts(actual_label)
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
predict_label = tf.argmax(y_conv, 1)
label_p, idx_p, count_p = tf.unique_with_counts(predict_label)
correct_prediction = tf.equal(predict_label, actual_label)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
correct_label = tf.boolean_mask(actual_label, correct_prediction)
label_c, idx_c, count_c = tf.unique_with_counts(correct_label)


# if model exists: restore it
# else: train a new model and save it
saver = tf.train.Saver()
model_name = "model_" + str(CLASS_NUM) + "class_" + folder
model = model_name + '/' + model_name + ".ckpt"
if not os.path.exists(model):
    sess.run(tf.global_variables_initializer())
    if not os.path.exists(model_name):
        os.makedirs(model_name)
    # with open('out.txt','a') as f:
    #     f.write(time.strftime('%Y-%m-%d %X',time.localtime()) + "\n")
    #     f.write('DATA_DIR: ' + DATA_DIR+ "\n")
    fig_f1 = np.zeros([TRAIN_ROUND+1])
    fig_acc =np.zeros([TRAIN_ROUND+1])

    for i in range(TRAIN_ROUND + 1):
        #print(np.shape(mnist))
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            # print(np.shape(batch[0]))
            # print(np.shape(batch[1]))
            train_accuracy,y_pred,y_true = sess.run([accuracy,predict_label,actual_label],feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            s1 = "step %d, train accuracy %g" % (i, train_accuracy)
            print(s1)
            # if i%2000 == 0:
            #     with open('out.txt','a') as f:
            #         f.write(s + "\n")
        sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        fig_acc[i] = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        y_pred, y_true = sess.run([predict_label, actual_label],
                                              feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        fig_f1[i]=metrics.f1_score(y_true,y_pred,average='macro')

    print(np.shape(mnist.test.images))
    print(np.shape(mnist.test.labels))
    s2= "test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    print(s2)

    save_path = saver.save(sess, model)
    print("Model saved in file:", save_path)

    # 绘制曲线
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    lns1 = ax1.plot(np.arange(TRAIN_ROUND+1), fig_acc, label="F1 Score")
    # 按一定间隔显示实现方法
    # ax2.plot(200 * np.arange(len(fig_acc)), fig_acc, 'r')
    lns2 = ax2.plot(np.arange(TRAIN_ROUND+1), fig_f1, 'r', label="Accuracy")
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Training Accuracy')
    ax2.set_ylabel('Training F1 Score')
    # 合并图例
    lns = lns1 + lns2
    labels = ["Accuracy", "F1 Score"]
    # labels = [l.get_label() for l in lns]
    plt.legend(lns, labels, loc=7)
    plt.show()



else:
    saver.restore(sess, model)
    print("Model restored: " + model)