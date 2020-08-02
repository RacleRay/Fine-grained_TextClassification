import tensorflow as tf
import numpy as np

x1 = ([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
x2 = ([[2, 1, 1], [2, 1, 1], [2, 1, 1]])
y1 = np.dot(x1, x2)
y2 = np.multiply(x1, x2)
print('1、np.dot\n', y1)
print('2、np.multiply\n', y2)

x3 = tf.constant([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
x4 = tf.constant([[2, 1, 1], [2, 1, 1], [2, 1, 1]])
y3 = tf.matmul(x3, x4)
y4 = tf.multiply(x3, x4)

with tf.Session() as sess:
    print('3、tf.matmul\n',sess.run(y3))
    print('4、tf.multiply\n',sess.run(y4))
    print('5、np.dot\n',sess.run(np.dot(x3,x4)))
    print('6、np.multiply\n',sess.run(np.multiply(x3,x4)))