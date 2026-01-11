import vgg
import PIL.Image
import numpy as np
import tensorflow as tf
import os

# === 新增：创建结果文件夹 ===
if not os.path.exists('./lapstyle_result_masked'):
    os.makedirs('./lapstyle_result_masked')

# === 新增：参数配置区域 ===
MASK_PATH = "data/masks/mask_cat.png"  # 请确保路径正确
BG_LAPLACE_WEIGHT = 0.05               # 背景 Laplace 权重 (0.05 = 允许背景狂野)
BG_CONTENT_WEIGHT = 0.1                # 背景 Content 权重 (0.1 = 允许背景变形)
STYLE_LAYER_WEIGHTS = [1.0, 2.0, 4.0, 10.0, 20.0] # 风格层权重 (浅->深)

def make_kernel(a):
  """Transform a 2D array into a convolution kernel"""
  a = np.asarray(a)
  a = a.reshape(list(a.shape) + [1, 1])
  return tf.constant(a, dtype=1)

def simple_conv(x, k):
  """A simplified 2D convolution operation"""
  x = tf.expand_dims(tf.expand_dims(x, 0), -1)
  # print (x.shape) # 注释掉以减少刷屏
  y = tf.nn.depthwise_conv2d(x[0,0], k, [1, 1, 1, 1], padding='SAME')
  return y[0, :, :, 0]

def laplace(x):
  """Compute the 2D laplacian of an array"""
  laplace_k = make_kernel([[0.5, 1.0, 0.5],
                           [1.0, -6., 1.0],
                           [0.5, 1.0, 0.5]])
  return simple_conv(x, laplace_k)

# === 新增：辅助函数，计算带蒙版的 Gram 矩阵 ===
def compute_masked_gram(feature_map, mask_resized):
    # 1. 应用蒙版：只保留感兴趣区域(这里是背景)的特征，其他置0
    masked_features = feature_map * mask_resized
    # 2. 归一化分母：使用 Mask 的有效面积
    num_channels = tf.cast(tf.shape(feature_map)[3], tf.float32)
    valid_pixels = tf.reduce_sum(mask_resized) + 1e-5
    # 3. 计算 Gram
    flat = tf.reshape(masked_features, [-1, tf.shape(feature_map)[3]])
    gram = tf.matmul(flat, flat, transpose_a=True) / (2.0 * num_channels * valid_pixels)
    return gram

# initializing the VGG network from pre trained downloaded Data
vgg = vgg.Vgg19()

# getting input images and resizing the style image to content image
content_image = np.asarray(PIL.Image.open("data/content-images/c1.jpg"), dtype=float) # 注意修改了文件名
img_width = content_image.shape[0]
img_height = content_image.shape[1]
style_image = np.asarray(PIL.Image.open("data/style-images/s2.jpg"))
style_image = tf.image.resize_images(style_image, size=[img_width, img_height])

# === 新增：加载并处理 Mask ===
# 读取为灰度(L)，缩放，归一化到 0-1
mask_np = np.asarray(PIL.Image.open(MASK_PATH).convert('L').resize((img_height, img_width)), dtype=float) / 255.0
mask_tensor = tf.reshape(tf.constant(mask_np, dtype=tf.float32), [1, img_width, img_height, 1])
# 定义前景(猫)和背景(天)的 Mask
mask_bg = mask_tensor        # 假设 mask 图片里背景是白色的(1.0)
mask_fg = 1.0 - mask_tensor  # 假设 mask 图片里背景是白色的，那猫就是黑的，取反

b = np.zeros(shape=[1, img_width, img_height, 3])
b[0] = content_image
input_var = tf.clip_by_value(tf.Variable(b, trainable=True, dtype=tf.float32), 0.0, 255.0)

# now building the pre trained vgg model graph for style transfer
vgg.build(input_var)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# getting all the style layers

# === 修改逻辑：手动计算每一层的 Masked Style Loss ===
# 注意：原代码的 target_style_matrix 是用 input_var 算的，这里为了准确，我们需要先算好全图风格及其 Gram

# 预先计算风格图的 Gram (全图)
# 为了不破坏原代码结构，我们这里用一种简便方法：手动定义 target gram 计算公式
# 注意：Style Loss = (Masked_Gram_Gen - Global_Gram_Style)^2
# 我们希望背景区域(Masked Gen) 长得像 整个风格图(Global Style)

# Layer 5
num_channels_5 = int(vgg.conv5_1.shape[3])
# 1. 计算 Target (全图风格)
flat_5 = tf.reshape(vgg.conv5_1, shape=[-1, int(vgg.conv5_1.shape[1]*vgg.conv5_1.shape[2]), num_channels_5])
gram_global_5 = tf.matmul(flat_5, flat_5/(2*num_channels_5*int(vgg.conv5_1.shape[1]*vgg.conv5_1.shape[2])), transpose_a=True)
target_style_matrix_5 = sess.run(gram_global_5, feed_dict={input_var: [style_image.eval(session=sess)]})
# 2. 计算 Generated (Masked 背景)
mask_bg_5 = tf.image.resize_nearest_neighbor(mask_bg, [int(vgg.conv5_1.shape[1]), int(vgg.conv5_1.shape[2])])
gram_gen_masked_5 = compute_masked_gram(vgg.conv5_1, mask_bg_5)
# 3. 计算 Loss (乘上深层权重 20.0)
style_loss_5 = tf.reduce_sum(tf.square(tf.subtract(gram_gen_masked_5, target_style_matrix_5))) * STYLE_LAYER_WEIGHTS[4]

# Layer 4
num_channels_4 = int(vgg.conv4_1.shape[3])
flat_4 = tf.reshape(vgg.conv4_1, shape=[-1, int(vgg.conv4_1.shape[1]*vgg.conv4_1.shape[2]), num_channels_4])
gram_global_4 = tf.matmul(flat_4, flat_4/(2.0*num_channels_4*int(vgg.conv4_1.shape[1]*vgg.conv4_1.shape[2])), transpose_a=True)
target_style_matrix_4 = sess.run(gram_global_4, feed_dict={input_var: [style_image.eval(session=sess)]})
mask_bg_4 = tf.image.resize_nearest_neighbor(mask_bg, [int(vgg.conv4_1.shape[1]), int(vgg.conv4_1.shape[2])])
gram_gen_masked_4 = compute_masked_gram(vgg.conv4_1, mask_bg_4)
style_loss_4 = tf.reduce_sum(tf.square(tf.subtract(gram_gen_masked_4, target_style_matrix_4))) * STYLE_LAYER_WEIGHTS[3]

# Layer 3
num_channels_3 = int(vgg.conv3_1.shape[3])
flat_3 = tf.reshape(vgg.conv3_1, shape=[-1, int(vgg.conv3_1.shape[1]*vgg.conv3_1.shape[2]), num_channels_3])
gram_global_3 = tf.matmul(flat_3, flat_3/(2.0*num_channels_3*int(vgg.conv3_1.shape[1]*vgg.conv3_1.shape[2])), transpose_a=True)
target_style_matrix_3 = sess.run(gram_global_3, feed_dict={input_var: [style_image.eval(session=sess)]})
mask_bg_3 = tf.image.resize_nearest_neighbor(mask_bg, [int(vgg.conv3_1.shape[1]), int(vgg.conv3_1.shape[2])])
gram_gen_masked_3 = compute_masked_gram(vgg.conv3_1, mask_bg_3)
style_loss_3 = tf.reduce_sum(tf.square(tf.subtract(gram_gen_masked_3, target_style_matrix_3))) * STYLE_LAYER_WEIGHTS[2]

# Layer 2
num_channels_2 = int(vgg.conv2_1.shape[3])
flat_2 = tf.reshape(vgg.conv2_1, shape=[-1, int(vgg.conv2_1.shape[1]*vgg.conv2_1.shape[2]), num_channels_2])
gram_global_2 = tf.matmul(flat_2, flat_2/(2.0*num_channels_2*int(vgg.conv2_1.shape[1]*vgg.conv2_1.shape[2])), transpose_a=True)
target_style_matrix_2 = sess.run(gram_global_2, feed_dict={input_var: [style_image.eval(session=sess)]})
mask_bg_2 = tf.image.resize_nearest_neighbor(mask_bg, [int(vgg.conv2_1.shape[1]), int(vgg.conv2_1.shape[2])])
gram_gen_masked_2 = compute_masked_gram(vgg.conv2_1, mask_bg_2)
style_loss_2 = tf.reduce_sum(tf.square(tf.subtract(gram_gen_masked_2, target_style_matrix_2))) * STYLE_LAYER_WEIGHTS[1]

# Layer 1
num_channels_1 = int(vgg.conv1_1.shape[3])
flat_1 = tf.reshape(vgg.conv1_1, shape=[-1, int(vgg.conv1_1.shape[1]*vgg.conv1_1.shape[2]), num_channels_1])
gram_global_1 = tf.matmul(flat_1, flat_1/(2.0*num_channels_1*int(vgg.conv1_1.shape[1]*vgg.conv1_1.shape[2])), transpose_a=True)
target_style_matrix_1 = sess.run(gram_global_1, feed_dict={input_var: [style_image.eval(session=sess)]})
mask_bg_1 = tf.image.resize_nearest_neighbor(mask_bg, [int(vgg.conv1_1.shape[1]), int(vgg.conv1_1.shape[2])])
gram_gen_masked_1 = compute_masked_gram(vgg.conv1_1, mask_bg_1)
style_loss_1 = tf.reduce_sum(tf.square(tf.subtract(gram_gen_masked_1, target_style_matrix_1))) * STYLE_LAYER_WEIGHTS[0]

# setting the loss variables for style
style_loss = (style_loss_1 + style_loss_2 + style_loss_3 + style_loss_4 + style_loss_5) / 5.0

# === 修改逻辑：带空间权重的 Content Loss ===
# 构建空间权重图：猫保持 1.0，天降为 0.1
# 注意：原代码是减法后平方，我们需要插入权重乘法

# Content Layer 4_2
mask_fg_4_2 = tf.image.resize_nearest_neighbor(mask_fg, [int(vgg.conv4_2.shape[1]), int(vgg.conv4_2.shape[2])])
mask_bg_4_2 = tf.image.resize_nearest_neighbor(mask_bg, [int(vgg.conv4_2.shape[1]), int(vgg.conv4_2.shape[2])])
spatial_weight_4 = mask_fg_4_2 * 1.0 + mask_bg_4_2 * BG_CONTENT_WEIGHT

target_content_matrix_1 = sess.run(vgg.conv4_2, feed_dict={input_var: [content_image]})
weight_1 = 2.0 * np.sqrt(int(vgg.conv4_2.shape[1])*int(vgg.conv4_2.shape[2])*int(vgg.conv4_2.shape[3]))
# 插入 spatial_weight_4
content_diff_1 = tf.square(tf.subtract(vgg.conv4_2, target_content_matrix_1)) * spatial_weight_4
content_loss_1 = tf.reduce_sum(content_diff_1) / weight_1

# Content Layer 1_2
mask_fg_1_2 = tf.image.resize_nearest_neighbor(mask_fg, [int(vgg.conv1_2.shape[1]), int(vgg.conv1_2.shape[2])])
mask_bg_1_2 = tf.image.resize_nearest_neighbor(mask_bg, [int(vgg.conv1_2.shape[1]), int(vgg.conv1_2.shape[2])])
spatial_weight_1 = mask_fg_1_2 * 1.0 + mask_bg_1_2 * BG_CONTENT_WEIGHT

weight_2 = 2.0 * np.sqrt(int(vgg.conv1_2.shape[1])*int(vgg.conv1_2.shape[2])*int(vgg.conv1_2.shape[3]))
target_content_matrix_2 = sess.run(vgg.conv1_2, feed_dict={input_var: [content_image]})
# 插入 spatial_weight_1
content_diff_2 = tf.square(tf.subtract(vgg.conv1_2, target_content_matrix_2)) * spatial_weight_1
content_loss_2 = tf.reduce_sum(content_diff_2) / weight_2

content_loss = (content_loss_1 + content_loss_2) / 2.0

# === 修改逻辑：带空间权重的 Laplace Loss ===
# 构建空间权重图 (原始尺寸)
lap_spatial_weight = mask_fg * 1.0 + mask_bg * BG_LAPLACE_WEIGHT

# Computting all of laplace losses
lap_1 = laplace(input_var)
target_laplace_1 = sess.run(lap_1, feed_dict={input_var: [content_image]})
# 插入 lap_spatial_weight
laplace_loss_1 = tf.reduce_mean(tf.square(lap_1 - target_laplace_1) * lap_spatial_weight[:,:,:,0])

lap_2 = laplace(tf.nn.pool(input_var, window_shape=[2, 2], pooling_type='AVG',padding='SAME'))
target_laplace_2 = sess.run(lap_2, feed_dict={input_var: [content_image]})
# Resize mask for pool2
weight_lap_2 = tf.image.resize_nearest_neighbor(lap_spatial_weight, [int(img_width/2), int(img_height/2)])[:,:,:,0]
laplace_loss_2 = tf.reduce_mean(tf.square(lap_2 - target_laplace_2) * weight_lap_2)

lap_3 = laplace(tf.nn.pool(input_var, window_shape=[4,4], pooling_type="AVG", padding="SAME"))
target_laplace_3 = sess.run(lap_3, feed_dict={input_var: [content_image]})
# Resize mask for pool4
weight_lap_3 = tf.image.resize_nearest_neighbor(lap_spatial_weight, [int(img_width/4), int(img_height/4)])[:,:,:,0]
laplace_loss_3 = tf.reduce_mean(tf.square(lap_3 - target_laplace_3) * weight_lap_3)

lap_4 = laplace(tf.nn.pool(input_var, window_shape=[8,8], pooling_type="AVG", padding="SAME"))
target_laplace_4 = sess.run(lap_4, feed_dict={input_var: [content_image]})
# Resize mask for pool8
weight_lap_4 = tf.image.resize_nearest_neighbor(lap_spatial_weight, [int(img_width/8), int(img_height/8)])[:,:,:,0]
laplace_loss_4 = tf.reduce_mean(tf.square(lap_4 - target_laplace_4) * weight_lap_4)

lap_5 = laplace(tf.nn.pool(input_var, window_shape=[10,10], pooling_type="AVG", padding="SAME"))
target_laplace_5 = sess.run(lap_5, feed_dict={input_var: [content_image]})
# Resize mask for pool10
weight_lap_5 = tf.image.resize_nearest_neighbor(lap_spatial_weight, [int(img_width/10), int(img_height/10)])[:,:,:,0]
laplace_loss_5 = tf.reduce_mean(tf.square(lap_5 - target_laplace_5) * weight_lap_5)

lap_6 = laplace(tf.nn.pool(input_var, window_shape=[16,16],pooling_type="AVG", padding="SAME"))
target_laplace_6 = sess.run(lap_6, feed_dict={input_var: [content_image]})
# Resize mask for pool16
weight_lap_6 = tf.image.resize_nearest_neighbor(lap_spatial_weight, [int(img_width/16), int(img_height/16)])[:,:,:,0]
laplace_loss_6 = tf.reduce_mean(tf.square(lap_6 - target_laplace_6) * weight_lap_6)

# === 修改逻辑：更新总权重系数以匹配新逻辑 ===
# 原代码 coefs: [100, 1, 1e8, ...]
# 新逻辑需要 Style 很大(因为做了Mask归一化)，Laplace 很大
coefs = [1e4, 1.0, 1e7, 1e7, 1e7, 1e7, 1e7, 1e7] # 这里的权重可以根据效果微调

total_loss = coefs[0]*style_loss + coefs[1]*content_loss + coefs[2]*laplace_loss_1 + coefs[3]*laplace_loss_2 + coefs[4]*laplace_loss_3 + coefs[5]*laplace_loss_4 + coefs[6]*laplace_loss_5 + coefs[7]*laplace_loss_6
train_op = tf.contrib.opt.ScipyOptimizerInterface(total_loss, method='L-BFGS-B', options={'maxiter': 1000})
sess.run(tf.global_variables_initializer())
a = []

_iter = 0

def callback(tl, cl, sl, ii):
    global _iter
    if _iter % 100 == 0:
        print('iter : %4d, ' % _iter, 'L_total : %g, L_content : %g, L_style : %g' % (tl, cl, sl))
        img1 = PIL.Image.fromarray(tf.cast(ii, dtype=tf.uint8).eval(session=sess)[0], 'RGB')
        img1.save('./lapstyle_result_masked/my'+str(_iter)+'.png')
        #img1.show()
    _iter += 1

train_op.minimize(sess, fetches=[total_loss, content_loss, style_loss, input_var], loss_callback=callback)

img = PIL.Image.fromarray(input_var.eval(session=sess)[0], 'RGB')
img.save('./lapstyle_result_masked/my.png')
#img.show()