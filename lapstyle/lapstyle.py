import vgg
import PIL.Image
import numpy as np
import tensorflow as tf


def make_kernel(a):
  """Transform a 2D array into a convolution kernel"""
  a = np.asarray(a)
  a = a.reshape(list(a.shape) + [1, 1])
  return tf.constant(a, dtype=1)

def simple_conv(x, k):
  """A simplified 2D convolution operation"""
  x = tf.expand_dims(tf.expand_dims(x, 0), -1)
  print (x.shape)
  y = tf.nn.depthwise_conv2d(x[0,0], k, [1, 1, 1, 1], padding='SAME')
  return y[0, :, :, 0]

def laplace(x):
  """Compute the 2D laplacian of an array"""
  laplace_k = make_kernel([[0.5, 1.0, 0.5],
                           [1.0, -6., 1.0],
                           [0.5, 1.0, 0.5]])
  return simple_conv(x, laplace_k)


# initializing the VGG network from pre trained downloaded Data
vgg = vgg.Vgg19()

# getting input images and resizing the style image to content image
content_image = np.asarray(PIL.Image.open("../data/content-images/c1.jpg"), dtype=float)
img_width = content_image.shape[0]
img_height = content_image.shape[1]
style_image = np.asarray(PIL.Image.open("../data/style-images/s2.jpg"))
style_image = tf.image.resize_images(style_image, size=[img_width, img_height])
b = np.zeros(shape=[1, img_width, img_height, 3])
b[0] = content_image
input_var = tf.clip_by_value(tf.Variable(b, trainable=True, dtype=tf.float32), 0.0, 255.0)

# now building the pre trained vgg model graph for style transfer
vgg.build(input_var)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# getting all the style layers

num_channels_5 = int(vgg.conv5_1.shape[3])
space_dim_5_1 = int(vgg.conv5_1.shape[1])
space_dim_5_2 = int(vgg.conv5_1.shape[2])
tensor_5 = tf.reshape(vgg.conv5_1, shape=[-1, space_dim_5_1*space_dim_5_2, num_channels_5])
gram_matrix_5 = tf.matmul(tensor_5, tensor_5/(2*num_channels_5*space_dim_5_1*space_dim_5_2), transpose_a=True)
target_style_matrix_5 = sess.run(gram_matrix_5, feed_dict={input_var: [style_image.eval(session=sess)]})
style_loss_5 = tf.reduce_sum(tf.square(tf.subtract(gram_matrix_5, target_style_matrix_5)), reduction_indices=[1, 2])

num_channels_4 = int(vgg.conv4_1.shape[3])
space_dim_4_1 = int(vgg.conv4_1.shape[1])
space_dim_4_2 = int(vgg.conv4_1.shape[2])
tensor_4 = tf.reshape(vgg.conv4_1, shape=[-1, space_dim_4_1*space_dim_4_2, num_channels_4])
gram_matrix_4 = tf.matmul(tensor_4, tensor_4/(2.0*num_channels_4*space_dim_4_1*space_dim_4_2), transpose_a=True)
target_style_matrix_4 = sess.run(gram_matrix_4, feed_dict={input_var: [style_image.eval(session=sess)]})
style_loss_4 = tf.reduce_sum(tf.square(tf.subtract(gram_matrix_4, target_style_matrix_4)), reduction_indices=[1, 2])

num_channels_3 = int(vgg.conv3_1.shape[3])
space_dim_3_1 = int(vgg.conv3_1.shape[1])
space_dim_3_2 = int(vgg.conv3_1.shape[2])
tensor_3 = tf.reshape(vgg.conv3_1, shape=[-1, space_dim_3_1*space_dim_3_2, num_channels_3])
gram_matrix_3 = tf.matmul(tensor_3, tensor_3/(2.0*num_channels_3*space_dim_3_1*space_dim_3_2), transpose_a=True)
target_style_matrix_3 = sess.run(gram_matrix_3, feed_dict={input_var: [style_image.eval(session=sess)]})
style_loss_3 = tf.reduce_sum(tf.square(tf.subtract(gram_matrix_3, target_style_matrix_3)), reduction_indices=[1, 2])

num_channels_2 = int(vgg.conv2_1.shape[3])
space_dim_2_1 = int(vgg.conv2_1.shape[1])
space_dim_2_2 = int(vgg.conv2_1.shape[2])
tensor_2 = tf.reshape(vgg.conv2_1, shape=[-1, space_dim_2_1*space_dim_2_2, num_channels_2])
gram_matrix_2 = tf.matmul(tensor_2, tensor_2/(2.0*num_channels_2*space_dim_2_1*space_dim_2_2), transpose_a=True)
target_style_matrix_2 = sess.run(gram_matrix_2, feed_dict={input_var: [style_image.eval(session=sess)]})
style_loss_2 = tf.reduce_sum(tf.square(tf.subtract(gram_matrix_2, target_style_matrix_2)), reduction_indices=[1, 2])

num_channels_1 = int(vgg.conv1_1.shape[3])
space_dim_1_1 = int(vgg.conv1_1.shape[1])
space_dim_1_2 = int(vgg.conv1_1.shape[2])
tensor_1 = tf.reshape(vgg.conv1_1, shape=[-1, space_dim_1_1*space_dim_1_2, num_channels_1])
gram_matrix_1 = tf.matmul(tensor_1, tensor_1/(2.0*num_channels_1*space_dim_1_1*space_dim_1_2), transpose_a=True)
target_style_matrix_1 = sess.run(gram_matrix_1, feed_dict={input_var: [style_image.eval(session=sess)]})
style_loss_1 = tf.reduce_sum(tf.square(tf.subtract(gram_matrix_1, target_style_matrix_1)), reduction_indices=[1, 2])

# setting the loss variables for style
style_loss = (style_loss_1 + style_loss_2 + style_loss_3 + style_loss_4 + style_loss_5) / 5.0

# getting the variables needed for optimization
target_content_matrix_1 = sess.run(vgg.conv4_2, feed_dict={input_var: [content_image]})
weight_1 = 2.0 * np.sqrt(int(vgg.conv4_2.shape[1])*int(vgg.conv4_2.shape[2])*int(vgg.conv4_2.shape[3]))
content_loss_1 = tf.reduce_sum(tf.square(tf.subtract(vgg.conv4_2, target_content_matrix_1)), reduction_indices=[1, 2, 3]) / weight_1
weight_2 = 2.0 * np.sqrt(int(vgg.conv1_2.shape[1])*int(vgg.conv1_2.shape[2])*int(vgg.conv1_2.shape[3]))
target_content_matrix_2 = sess.run(vgg.conv1_2, feed_dict={input_var: [content_image]})
content_loss_2 = tf.reduce_sum(tf.square(tf.subtract(vgg.conv1_2, target_content_matrix_2)), reduction_indices=[1, 2, 3]) / weight_2
content_loss = (content_loss_1 + content_loss_2) / 2.0

# Computting all of laplace losses
lap_1 = laplace(input_var)
target_laplace_1 = sess.run(lap_1, feed_dict={input_var: [content_image]})
laplace_loss_1 = tf.reduce_mean(tf.square(lap_1 - target_laplace_1))

lap_2 = laplace(tf.nn.pool(input_var, window_shape=[2, 2], pooling_type='AVG',padding='SAME'))
target_laplace_2 = sess.run(lap_2, feed_dict={input_var: [content_image]})
laplace_loss_2 = tf.reduce_mean(tf.square(lap_2 - target_laplace_2))

lap_3 = laplace(tf.nn.pool(input_var, window_shape=[4,4], pooling_type="AVG", padding="SAME"))
target_laplace_3 = sess.run(lap_3, feed_dict={input_var: [content_image]})
laplace_loss_3 = tf.reduce_mean(tf.square(lap_3 - target_laplace_3))

lap_4 = laplace(tf.nn.pool(input_var, window_shape=[8,8], pooling_type="AVG", padding="SAME"))
target_laplace_4 = sess.run(lap_4, feed_dict={input_var: [content_image]})
laplace_loss_4 = tf.reduce_mean(tf.square(lap_4 - target_laplace_4))

lap_5 = laplace(tf.nn.pool(input_var, window_shape=[10,10], pooling_type="AVG", padding="SAME"))
target_laplace_5 = sess.run(lap_5, feed_dict={input_var: [content_image]})
laplace_loss_5 = tf.reduce_mean(tf.square(lap_5 - target_laplace_5))

lap_6 = laplace(tf.nn.pool(input_var, window_shape=[16,16],pooling_type="AVG", padding="SAME"))
target_laplace_6 = sess.run(lap_6, feed_dict={input_var: [content_image]})
laplace_loss_6 = tf.reduce_mean(tf.square(lap_6 - target_laplace_6))

coefs = [100, 1, 100000000, 100000000, 100000000, 100000000, 100000000, 100000000]
total_loss = coefs[0]*style_loss + coefs[1]*content_loss + coefs[2]*laplace_loss_1 + coefs[3]*laplace_loss_2 + coefs[4]*laplace_loss_3 + coefs[5]*laplace_loss_4 + coefs[6]*laplace_loss_5 + coefs[7]*laplace_loss_6
train_op = tf.contrib.opt.ScipyOptimizerInterface(total_loss, method='L-BFGS-B', options={'maxiter': 1000})
sess.run(tf.global_variables_initializer())
a = []

_iter = 0


def callback(tl, cl, sl, ii):
    global _iter
    print('iter : %4d, ' % _iter, 'L_total : %g, L_content : %g, L_style : %g' % (tl, cl, sl))
    if _iter % 100 == 0:
        img1 = PIL.Image.fromarray(tf.cast(ii, dtype=tf.uint8).eval(session=sess)[0], 'RGB')
        img1.save('./my'+str(_iter)+'.png')
        img1.show()
    _iter += 1


train_op.minimize(sess, fetches=[total_loss, content_loss, style_loss, input_var], loss_callback=callback)

img = PIL.Image.fromarray(input_var.eval(session=sess)[0], 'RGB')
img.save('./my.png')
img.show()
