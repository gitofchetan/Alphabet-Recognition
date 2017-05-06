from __future__ import print_function

import csv
import numpy as np

f1 = 'test_image_matrix_tr1.csv';
f2 = 'test_lable_onehot1.csv';
f3 = 'training_image_matrix_tr1.csv';
f4 = 'training_lable_onehot1.csv';
f5 = 'validation_image_matrix_tr1.csv';
f6 = 'validation_lable_onehot1.csv';

def loadCsv(filename):
	lines = csv.reader(open(filename, "rt"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
#		print dataset[i]
	return dataset

te_labels = loadCsv(f2)
tr_labels = loadCsv(f4)
va_labels = loadCsv(f6)
te_image = loadCsv(f1)
tr_image = loadCsv(f3)
va_image = loadCsv(f5)

num = len(tr_labels)

import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 500
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 30 # 1st layer number of features
n_hidden_2 = 15 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 26 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
#1 Hidden Layer
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
#1 Hidden Layer
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]),name='w1'),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]),name='w3')
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]),name='b1'),
    'out': tf.Variable(tf.random_normal([n_classes]),name='b3')
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

total_batch = int(num/batch_size)
image = []
label = []
n=0
#Matrix = [[0 for x in range(w)] for y in range(h)] 
for i in range(total_batch):
    image.append(tr_image[n:n+100][:])
    label.append(tr_labels[n:n+100][:])
    n+=100
    
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        # Loop over all batches
        for i in range(total_batch):
#            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x, batch_y = image[i],label[i]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!") 
#    save_path = saver.save(sess, "/tmp/model.csv")
#    print("Model saved in file: %s" % save_path)

    v1 = sess.run(weights["h1"])
    v2 = sess.run(weights["out"])
    v3 = sess.run(biases["b1"])
    v4 = sess.run(biases["out"])

    v1.tofile('weight_layer1.txt', sep=' ')
    v2.tofile('weight_layer3.txt', sep=' ')
    v3.tofile('bias_layer1.txt', sep=' ')
    v4.tofile('bias_layer3.txt', sep=' ')


    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: va_image, y: va_labels}))

