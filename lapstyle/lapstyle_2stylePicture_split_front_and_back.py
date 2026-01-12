import numpy as np
import cv2
import os
import vgg
import PIL.Image
import tensorflow as tf

# ==========================================
# 第一部分：生成双风格图的辅助函数
# ==========================================

def get_color_stats(img, mask):
    """计算图片在 mask 区域内的均值和标准差"""
    # mask: 0 或 255
    # 提取有效像素
    valid_pixels = img[mask > 128] # 阈值提取
    if len(valid_pixels) == 0:
        return np.zeros(3), np.zeros(3)
    
    #计算均值和标准差
    mean = np.mean(valid_pixels, axis=0)
    std = np.std(valid_pixels, axis=0)
    return mean, std

def apply_color_transfer(style_img, mean_target, std_target):
    """将目标均值/方差应用到 style_img"""
    style_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2LAB).astype(float)
    
    mean_style = np.mean(style_img, axis=(0,1))
    std_style = np.std(style_img, axis=(0,1))
    
    # 避免除以0
    std_style[std_style == 0] = 1e-5
    
    # 转换
    # (x - mean) * (std_tgt / std_src) + mean_tgt
    for i in range(3):
        # 这里只迁移均值，保留风格图原本的对比度(std)，以保留漩涡纹理
        style_img[:,:,i] = (style_img[:,:,i] - mean_style[i]) + mean_target[i]
        
    style_img = np.clip(style_img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(style_img, cv2.COLOR_LAB2BGR)

# === 预处理流程：生成风格图 ===
# 1. 配置路径
content_path = "data/content-images/c1.jpg"
style_path = "data/style-images/s2.jpg"
mask_path = "data/masks/mask_cat.png"
output_dir = "data/dual_styles"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. 读取图片 (OpenCV 读取默认为 BGR)
content = cv2.imread(content_path)
style = cv2.imread(style_path)
mask = cv2.imread(mask_path, 0) # 单通道读取

if content is None or style is None or mask is None:
    print("错误：无法读取图片，请检查路径！")
    exit()

# 缩放 mask 以匹配 content (防止尺寸不一)
mask = cv2.resize(mask, (content.shape[1], content.shape[0]), interpolation=cv2.INTER_NEAREST)

# 3. 转换 Content 到 LAB 空间提取统计信息
content_lab = cv2.cvtColor(content, cv2.COLOR_BGR2LAB).astype(float)

# 后景统计 (白色，Mask = 255)
mean_bg, std_bg = get_color_stats(content_lab, mask)
print(f"后景均值(LAB): {mean_bg}")

# 前景统计 (Mask = 0 -> invert mask)
mask_fg_cv = cv2.bitwise_not(mask)
mean_fg, std_fg = get_color_stats(content_lab, mask_fg_cv)
print(f"前景均值(LAB): {mean_fg}")

# 4. 生成两张风格图
print("正在生成前景风格图...")
style_fg = apply_color_transfer(style.copy(), mean_fg, std_fg)
cv2.imwrite(os.path.join(output_dir, "style_fg.jpg"), style_fg)

print("正在生成背景风格图...")
style_bg = apply_color_transfer(style.copy(), mean_bg, std_bg)
cv2.imwrite(os.path.join(output_dir, "style_bg.jpg"), style_bg)

print("完成！风格图已生成，开始训练流程...")


# ==========================================
# 第二部分：TensorFlow 训练主流程
# ==========================================

# 创建结果文件夹
if not os.path.exists('./lapstyle_result_dual'):
    os.makedirs('./lapstyle_result_dual')

# ================= 配置区域 =================
MASK_PATH = mask_path # 复用上面的路径
CONTENT_PATH = content_path

# 【重要】这里改为加载刚才生成的两张分色图
STYLE_FG_PATH = "data/dual_styles/style_fg.jpg" # 前景风格(猫色)
STYLE_BG_PATH = "data/dual_styles/style_bg.jpg" # 背景风格(天色)

# 权重配置
BG_LAPLACE_WEIGHT = 0.05 
BG_CONTENT_WEIGHT = 0.1  
# 风格权重系数 (浅->深)
STYLE_LAYER_WEIGHTS = [1.0, 2.0, 4.0, 10.0, 20.0] 
# ===========================================

def make_kernel(a):
    a = np.asarray(a)
    a = a.reshape(list(a.shape) + [1, 1])
    return tf.constant(a, dtype=1)

def simple_conv(x, k):
    """
    正确的 2D 卷积操作，保持空间维度 [1, H, W, C]
    x: [1, H, W, C]
    k: [3, 3, 1, 1]
    """
    # 获取输入通道数 (通常是 3)
    num_channels = int(x.get_shape()[-1])
    
    # 将 1通道的 kernel 复制平铺成 3通道: [3, 3, C, 1]
    # 这样才能对 RGB 三个通道分别做处理
    k_tiled = tf.tile(k, [1, 1, num_channels, 1])
    
    # 进行 Depthwise 卷积，保持 padding='SAME' 以维持尺寸
    y = tf.nn.depthwise_conv2d(x, k_tiled, strides=[1, 1, 1, 1], padding='SAME')
    return y

def laplace(x):
    laplace_k = make_kernel([[0.5, 1.0, 0.5],
                             [1.0, -6., 1.0],
                             [0.5, 1.0, 0.5]])
    return simple_conv(x, laplace_k)

def compute_masked_gram(feature_map, mask_resized):
    # 计算特定 Mask 区域内的 Gram 矩阵
    masked_features = feature_map * mask_resized
    num_channels = tf.cast(tf.shape(feature_map)[3], tf.float32)
    valid_pixels = tf.reduce_sum(mask_resized) + 1e-5
    flat = tf.reshape(masked_features, [-1, tf.shape(feature_map)[3]])
    gram = tf.matmul(flat, flat, transpose_a=True) / (2.0 * num_channels * valid_pixels)
    return gram

# 辅助函数：计算全图 Gram (用于预计算 Target)
def compute_global_gram(feature_map):
    num_channels = tf.cast(tf.shape(feature_map)[3], tf.float32)
    num_pixels = tf.cast(tf.reduce_prod(tf.shape(feature_map)[1:3]), tf.float32)
    flat = tf.reshape(feature_map, [-1, tf.shape(feature_map)[3]])
    gram = tf.matmul(flat, flat, transpose_a=True) / (2.0 * num_channels * num_pixels)
    return gram

vgg = vgg.Vgg19()

# 1. 加载 Content
content_img_np = np.asarray(PIL.Image.open(CONTENT_PATH).convert('RGB'), dtype=float)
img_width = content_img_np.shape[0]
img_height = content_img_np.shape[1]

# 2. 加载两张 Style 图片
style_fg_np = np.asarray(PIL.Image.open(STYLE_FG_PATH).convert('RGB'))
style_fg_resized = tf.image.resize_images(style_fg_np, size=[img_width, img_height])

style_bg_np = np.asarray(PIL.Image.open(STYLE_BG_PATH).convert('RGB'))
style_bg_resized = tf.image.resize_images(style_bg_np, size=[img_width, img_height])

# 3. 加载 Mask
mask_np = np.asarray(PIL.Image.open(MASK_PATH).convert('L').resize((img_height, img_width)), dtype=float) / 255.0
mask_tensor = tf.reshape(tf.constant(mask_np, dtype=tf.float32), [1, img_width, img_height, 1])

# 假设 Mask: 1=前景(猫), 0=背景(天) -- 根据实际情况调整
# 如果原图是白猫(1)，则 mask_fg = mask, mask_bg = 1-mask
mask_bg = mask_tensor 
mask_fg = 1 - mask_tensor

b = np.zeros(shape=[1, img_width, img_height, 3])
b[0] = content_img_np

# 1. 先定义真正的变量 (Variable)
raw_image_var = tf.Variable(b, trainable=True, dtype=tf.float32)

# 2. 再定义用于网络的裁剪版 (Tensor)
input_var = tf.clip_by_value(raw_image_var, 0.0, 255.0)

vgg.build(input_var)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# ================= 预计算两种 Target Grams =================
layers = [vgg.conv1_1, vgg.conv2_1, vgg.conv3_1, vgg.conv4_1, vgg.conv5_1]

# 1. 计算【前景风格】的目标 Gram (style_fg_resized)
target_grams_fg = []
style_fg_val = sess.run(style_fg_resized)
sess.run(raw_image_var.assign(style_fg_val.reshape(1, img_width, img_height, 3)))# 临时喂入
for layer in layers:
    target_grams_fg.append(sess.run(compute_global_gram(layer)))

# 2. 计算【背景风格】的目标 Gram (style_bg_resized)
target_grams_bg = []
style_bg_val = sess.run(style_bg_resized)
sess.run(raw_image_var.assign(style_bg_val.reshape(1, img_width, img_height, 3)))
for layer in layers:
    target_grams_bg.append(sess.run(compute_global_gram(layer)))

# 恢复 Content Image 到 input_var
sess.run(raw_image_var.assign(b))

# ================= 计算分区域 Style Loss =================
# 策略：Loss = (前景Gram - 前景Target)^2 + (背景Gram - 背景Target)^2

style_loss = 0.0

# Layer 5 (Conv5_1)
h, w = int(vgg.conv5_1.shape[1]), int(vgg.conv5_1.shape[2])
m_bg_5 = tf.image.resize_nearest_neighbor(mask_bg, [h, w])
m_fg_5 = tf.image.resize_nearest_neighbor(mask_fg, [h, w])

# 前景 Loss (去匹配 style_fg)
gram_gen_fg_5 = compute_masked_gram(vgg.conv5_1, m_fg_5)
loss_fg_5 = tf.reduce_sum(tf.square(gram_gen_fg_5 - target_grams_fg[4]))

# 背景 Loss (去匹配 style_bg)
gram_gen_bg_5 = compute_masked_gram(vgg.conv5_1, m_bg_5)
loss_bg_5 = tf.reduce_sum(tf.square(gram_gen_bg_5 - target_grams_bg[4]))

style_loss_5 = (loss_fg_5 + loss_bg_5) * STYLE_LAYER_WEIGHTS[4]


# Layer 4 (Conv4_1)
h, w = int(vgg.conv4_1.shape[1]), int(vgg.conv4_1.shape[2])
m_bg_4 = tf.image.resize_nearest_neighbor(mask_bg, [h, w])
m_fg_4 = tf.image.resize_nearest_neighbor(mask_fg, [h, w])

gram_gen_fg_4 = compute_masked_gram(vgg.conv4_1, m_fg_4)
loss_fg_4 = tf.reduce_sum(tf.square(gram_gen_fg_4 - target_grams_fg[3]))

gram_gen_bg_4 = compute_masked_gram(vgg.conv4_1, m_bg_4)
loss_bg_4 = tf.reduce_sum(tf.square(gram_gen_bg_4 - target_grams_bg[3]))

style_loss_4 = (loss_fg_4 + loss_bg_4) * STYLE_LAYER_WEIGHTS[3]


# Layer 3 (Conv3_1)
h, w = int(vgg.conv3_1.shape[1]), int(vgg.conv3_1.shape[2])
m_bg_3 = tf.image.resize_nearest_neighbor(mask_bg, [h, w])
m_fg_3 = tf.image.resize_nearest_neighbor(mask_fg, [h, w])

gram_gen_fg_3 = compute_masked_gram(vgg.conv3_1, m_fg_3)
loss_fg_3 = tf.reduce_sum(tf.square(gram_gen_fg_3 - target_grams_fg[2]))

gram_gen_bg_3 = compute_masked_gram(vgg.conv3_1, m_bg_3)
loss_bg_3 = tf.reduce_sum(tf.square(gram_gen_bg_3 - target_grams_bg[2]))

style_loss_3 = (loss_fg_3 + loss_bg_3) * STYLE_LAYER_WEIGHTS[2]


# Layer 2 (Conv2_1)
h, w = int(vgg.conv2_1.shape[1]), int(vgg.conv2_1.shape[2])
m_bg_2 = tf.image.resize_nearest_neighbor(mask_bg, [h, w])
m_fg_2 = tf.image.resize_nearest_neighbor(mask_fg, [h, w])

gram_gen_fg_2 = compute_masked_gram(vgg.conv2_1, m_fg_2)
loss_fg_2 = tf.reduce_sum(tf.square(gram_gen_fg_2 - target_grams_fg[1]))

gram_gen_bg_2 = compute_masked_gram(vgg.conv2_1, m_bg_2)
loss_bg_2 = tf.reduce_sum(tf.square(gram_gen_bg_2 - target_grams_bg[1]))

style_loss_2 = (loss_fg_2 + loss_bg_2) * STYLE_LAYER_WEIGHTS[1]


# Layer 1 (Conv1_1)
h, w = int(vgg.conv1_1.shape[1]), int(vgg.conv1_1.shape[2])
m_bg_1 = tf.image.resize_nearest_neighbor(mask_bg, [h, w])
m_fg_1 = tf.image.resize_nearest_neighbor(mask_fg, [h, w])

gram_gen_fg_1 = compute_masked_gram(vgg.conv1_1, m_fg_1)
loss_fg_1 = tf.reduce_sum(tf.square(gram_gen_fg_1 - target_grams_fg[0]))

gram_gen_bg_1 = compute_masked_gram(vgg.conv1_1, m_bg_1)
loss_bg_1 = tf.reduce_sum(tf.square(gram_gen_bg_1 - target_grams_bg[0]))

style_loss_1 = (loss_fg_1 + loss_bg_1) * STYLE_LAYER_WEIGHTS[0]

# 总风格损失
style_loss = (style_loss_1 + style_loss_2 + style_loss_3 + style_loss_4 + style_loss_5) / 5.0


# --- B. Content Loss (保持不变，使用空间权重) ---
def weighted_content_loss(layer, target):
    h, w = int(layer.shape[1]), int(layer.shape[2])
    spatial_weight = tf.image.resize_nearest_neighbor(mask_fg, [h, w]) * 1.0 + \
                     tf.image.resize_nearest_neighbor(mask_bg, [h, w]) * BG_CONTENT_WEIGHT
    diff = tf.square(layer - target)
    weighted_diff = diff * spatial_weight
    loss = tf.reduce_sum(weighted_diff) / (2.0 * np.sqrt(h * w * int(layer.shape[3])))
    return loss

target_content_4_2 = sess.run(vgg.conv4_2, feed_dict={input_var: b})
target_content_1_2 = sess.run(vgg.conv1_2, feed_dict={input_var: b})
content_loss = (weighted_content_loss(vgg.conv4_2, target_content_4_2) + 
                weighted_content_loss(vgg.conv1_2, target_content_1_2)) / 2.0


# --- C. Laplace Loss (修复了维度匹配问题) ---
laplace_loss = 0.0
# 确保权重图是 [1, H, W, 1]
lap_spatial_weight = mask_fg * 1.0 + mask_bg * BG_LAPLACE_WEIGHT 
pool_sizes = [1, 2, 4, 8, 10, 16]

for p_size in pool_sizes:
    if p_size == 1:
        pool_input = input_var
        current_weight = lap_spatial_weight
    else:
        # 【关键修复】这里必须添加 strides=[p_size, p_size]，确保图片和权重同步缩小
        pool_input = tf.nn.pool(input_var, window_shape=[p_size, p_size], pooling_type='AVG', 
                                padding='SAME', strides=[p_size, p_size])
        
        # Resize nearest neighbor 会保留 [1, H, W, 1] 的维度
        current_weight = tf.image.resize_nearest_neighbor(lap_spatial_weight, 
                                                          [int(img_width/p_size), int(img_height/p_size)])

    lap = laplace(pool_input)
    target_lap = sess.run(lap, feed_dict={raw_image_var: b}) # 注意这里 feed 给 raw_image_var
    
    diff = tf.square(lap - target_lap)
    
    # 修复后的乘法：[1,H,W,3] * [1,H,W,1] -> 自动广播
    weighted_diff = diff * current_weight 
    
    laplace_loss += tf.reduce_mean(weighted_diff)

# ================= 训练循环 =================
coefs = [1e4, 1.0, 1e7, 1e7, 1e7, 1e7, 1e7, 1e7]
total_loss = coefs[0]*style_loss + coefs[1]*content_loss + coefs[2]*laplace_loss

train_op = tf.contrib.opt.ScipyOptimizerInterface(total_loss, method='L-BFGS-B', options={'maxiter': 1000})
sess.run(tf.global_variables_initializer())

_iter = 0
def callback(tl, cl, sl, ii):
    global _iter
    if _iter % 100 == 0:
        print('iter : %4d, ' % _iter, 'L_total : %g, L_content : %g, L_style : %g' % (tl, cl, sl))
        img1 = PIL.Image.fromarray(tf.cast(ii, dtype=tf.uint8).eval(session=sess)[0], 'RGB')
        img1.save('./lapstyle_result_dual/my'+str(_iter)+'.png')
    _iter += 1

train_op.minimize(sess, fetches=[total_loss, content_loss, style_loss, input_var], loss_callback=callback)

img = PIL.Image.fromarray(input_var.eval(session=sess)[0], 'RGB')
img.save('./lapstyle_result_dual/my.png')