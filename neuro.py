import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, recall_score, classification_report
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve, auc

y_b = label_binarize(y, classes=['50005','52730','51381','50260','50000','50088','51380',
                                 '52490','56710','54860','52065','50961','52883','54770',
                                 '56490'])
X_train, X_test, y_train, y_test = train_test_split(X, y_b, test_size=0.3, random_state = 0)
x = tf.placeholder(tf.float32, [None, 1781])
W1 = tf.Variable(tf.truncated_normal([1781,500], stddev=0.1))
b1 = tf.Variable(tf.zeros([500]))
layer1 = tf.nn.tanh(tf.matmul(x, W1) + b1)

W2 = tf.Variable(tf.truncated_normal([500,11], stddev=0.1))
b2 = tf.Variable(tf.zeros([11]))

y = tf.nn.softmax(tf.matmul(layer1, W2) + b2)
y_ = tf.placeholder(tf.float32, [None, 11])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits = y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession
tf.global_variables_initializer().run()

for each in range(10):
    sess.run(train_step, feed_dict={x: X_train, y_: y_train})