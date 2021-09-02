# -*- coding: utf-8 -*-


from __future__ import absolute_import, division, print_function

import tensorflow as tf

import numpy as np

"""MNIST data set. MNIST data is a collection of hand-written digits that contains 60,000 examples for training and 10,000 examples for testing. The digits have been size-normalized and centered in a fixed-size image (28x28 pixels) with values from 0 to 255. 

Next for each image we will:

1) converted it to float32

2) normalized to [0, 1]

3) flattened to a 1-D array of 784 features (28*28).

#Step 2: Loading and Preparing the MNIST Data Set
"""

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Convert to float32.

x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)

# Flatten images to 1-D vector of 784 features (28*28).
num_features=784
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])


# Normalize images value from [0, 255] to [0, 1].
x_train, x_test = x_train / 255., x_test / 255.

"""#Step 3: Setting Up Hyperparameters and Data Set Parameters

Initialize the model parameters. 

num_classes denotes the number of outputs, which is 10, as we have digits from 0 to 9 in the data set. 

num_features defines the number of input parameters, and we store 784 since each image contains 784 pixels.
"""

# MNIST dataset parameters.

num_classes = 10 # 0 to 9 digits

num_features = 784 # 28*28

# Training parameters.

learning_rate = 0.01

training_steps = 1000

batch_size = 256

display_step = 50

"""#Step 4: Shuffling and Batching the Data


"""

# Use tf.data API to shuffle and batch data.
train_data=tf.data.Dataset.from_tensor_slices((x_train,y_train))

train_data=train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

"""#Step 5: Initializing Weights and Biases

We now initialize the weights vector and bias vector with ones and zeros.
"""

# Weight of shape [784, 10], the 28*28 image features, and a total number of classes.

W = tf.Variable(tf.ones([num_features, num_classes]), name="weight")

# Bias of shape [10], the total number of classes.
b = tf.Variable(tf.zeros([num_classes]), name="bias")

"""#Step 6: Defining Logistic Regression and Cost Function

We define the logistic_regression function below, which converts the inputs into a probability distribution proportional to the exponents of the inputs using the softmax function. The softmax function, which is implemented using the function tf.nn.softmax, also makes sure that the sum of all the inputs equals one.
"""

# Logistic regression (Wx + b).

def logistic_regression(x):    
    return tf.nn.softmax(tf.matmul(x, W) + b)
    

# Cross-Entropy loss function.

def cross_entropy(y_pred, y_true):

    # Encode label to a one hot vector.
    y_true = tf.one_hot(y_true, depth=num_classes)
    

    # Clip prediction values to avoid log(0) error.
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    

    # Compute cross-entropy.
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))

"""#Step 7: Defining Optimizers and Accuracy Metrics

"""

# Accuracy metric.

def accuracy(y_pred, y_true):

  # Predicted class is the index of the highest score in prediction vector (i.e. argmax).

  correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))

  return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

"""#Step 8: Optimization Process and Updating Weights and Biases
Now we define the run_optimization() method where we update the weights of our model. We calculate the predictions using the logistic_regression(x) method by taking the inputs and find out the loss generated by comparing the predicted value and the original value present in the data set. Next, we compute the gradients using and update the weights of the model with our stochastic gradient descent optimizer.
"""

# Optimization process. 

def run_optimization(x, y):

# Wrap computation inside a GradientTape for automatic differentiation.
  with tf.GradientTape() as g:
    pred = logistic_regression(x)
    loss = cross_entropy(pred, y)

    # Compute gradients.
  gradients = g.gradient(loss, [W, b])
  optimizer = tf.optimizers.SGD(learning_rate)
    
    # Update W and b following gradients.

  optimizer.apply_gradients(zip(gradients, [W, b]))

"""#Step 9: The Training Loop"""

# Run training for the given number of steps.

for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):

    # Run the optimization to update W and b values.

    run_optimization(batch_x, batch_y)

    

    if step % display_step == 0:

        pred = logistic_regression(batch_x)
       
        loss = cross_entropy(pred, batch_y)
        
        acc = accuracy(pred, batch_y)
        
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

"""#Step 10: Testing Model Accuracy Using the Test Data

Finally, we check the model accuracy by sending the test data set into our model and compute the accuracy using the accuracy function that we defined earlier.
"""

# Test model on validation set.
pred = logistic_regression(x_test)

print("Test Accuracy: %f" % accuracy(pred, y_test))