import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

LR = 0.01
epochs = 1000

# Creating dataset
X = np.linspace(-3,3,100)
np.random.seed(6)
Y = np.sin(X)+np.random.uniform(-0.5,0.5,100)

p = plt.plot(X,Y)
plt.axis([-4,4,-2,2])

# Experiment 1: Linear Regression - To find the best value of x

# x & y are fed to the session as placeholder
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
w = tf.Variable(tf.random_normal([1]),name="weights")
b = tf.Variable(tf.random_normal([1]), name='bias')

pred = tf.add(tf.multiply(x,w),b)
error = tf.square(pred - y)

opt = tf.train.GradientDescentOptimizer(LR).minimize(error)
init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)
    for epoch in range(epochs):
        for(xp,yp) in zip(X,Y):
            sess.run(opt,{x:xp,y:yp}) # Training
        cost = sess.run(error,{x:X,y:Y}) # Cost function
        if epoch % 10 == 0:
            plt.axis([-4,4,-2.0,2.0])
            plt.plot(X,pred.eval(feed_dict={x: X}, session=sess),
                     'b', alpha=epoch / epochs)
plt.show()
