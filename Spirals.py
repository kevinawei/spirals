import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# Read the data from the file
# I converted the txt file to a csv file to make it easier to parse

def read_dataset():
    # Turn data into a pandas dataset
    data = pd.read_csv('spiral.csv')
    X = data[data.columns[0:2]].values
    y = data[data.columns[2]]

    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    Y = one_hot_encode(y)
    return X, Y


def one_hot_encode(labels) :
        n_labels = len(labels)
        n_unique_labels = len(np.unique(labels))
        one_hot_encode = np.zeros((n_labels, n_unique_labels))
        one_hot_encode[np.arange(n_labels), labels] = 1
        return one_hot_encode


# Convert data into training and testing datasets after shuffling
X, Y = read_dataset()
X, Y = shuffle(X, Y, random_state=1)
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=.20, random_state=415)

# Parameters
model_path = "C:/Users/Kevin/Desktop/Model/Spiral_Model"
learning_rate = .001
epochs = 2000


# Network params including number of neurons for each layer
n_class = 2
n_layer_1 = 220
n_layer_2 = 110
n_input = X.shape[1]

# Graph Inputs
x = tf.placeholder(tf.float32, [None, n_input])
y_ = tf.placeholder(tf.float32, [None, n_class])
W = tf.Variable(tf.zeros([n_input, n_class]))
b = tf.Variable(tf.zeros([n_class]))

weights = {
    'w1': tf.Variable(tf.random_normal([n_input, n_layer_1])),
    'w2': tf.Variable(tf.random_normal([n_layer_1, n_layer_2])),
    'out': tf.Variable(tf.random_normal([n_layer_2, n_class]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_layer_1])),
    'b2': tf.Variable(tf.random_normal([n_layer_2])),
    'out': tf.Variable(tf.random_normal([n_class]))
}


# 2 layers, both using relu
def multilayer_perceptron():
    layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


logits = multilayer_perceptron()
cost_history = []

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
training_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)
init = tf.initialize_all_variables()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    # Training
    for epoch in range(epochs):
        sess.run(training_step, feed_dict={x: train_x, y_: train_y})
        cost = sess.run(loss_op, feed_dict={x: train_x, y_: train_y})
        cost_history = np.append(cost_history, cost)
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_,1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        pred_y = sess.run(logits, feed_dict={x: test_x})
        accuracy = (sess.run(accuracy, feed_dict={x: train_x, y_: train_y}))

        print('epoch: ', epoch, '-', 'cost: ', cost, "Accuracy: ", accuracy)

    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)
    saver.restore(sess, model_path)

    prediction = tf.argmax(logits, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Model is tested against all 194 points
    # All points are graphed according to their classification by the model
    print("TEST")
    accuracy_count = 0
    neg_coords_x = []
    neg_coords_y = []
    pos_coords_x = []
    pos_coords_y = []
    for i in range(0,193):
        prediction_run = sess.run(prediction, feed_dict={x: X[i].reshape(1, 2)})
        accuracy_run = sess.run(accuracy, feed_dict={x: X[i].reshape(1, 2), y_: Y[i].reshape(1, 2)})
        accuracy_count += accuracy_run
        if prediction_run == 0.0:

            neg_coords_x.append(X[i][0])
            neg_coords_y.append(X[i][1])
        if prediction_run == 1.0:
            pos_coords_x.append(X[i][0])
            pos_coords_y.append(X[i][1])

        print("Original Class: ", Y[i], " Predicted Values : ", prediction_run, "Accuracy: ", accuracy_run )
    print("Percentage of correct guesses: ", accuracy_count/194)

    plt.scatter(neg_coords_x, neg_coords_y)
    plt.scatter(pos_coords_x, pos_coords_y)
    plt.savefig('spirals.png')
