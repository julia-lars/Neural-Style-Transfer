import numpy as np
import cv2
import os
import vgg
import PIL.Image
import tensorflow as tf

# ==========================================
# 0. 全局配置与辅助函数
# ==========================================

# 风格图列表 (请确保这些文件都在 data/style-images/ 下)
STYLE_IMAGES_LIST = ['s1.jpg', 's2.jpg', 's3.png', 's4.png', 's5.png']

# 固定的内容图和Mask路径
CONTENT_PATH = "data/content-images/c1.jpg"
MASK_PATH = "data/masks/mask_cat.png"

# 总输出大文件夹
MAIN_OUTPUT_DIR = "data/batch_results_dual"

if not os.path.exists(MAIN_OUTPUT_DIR):
    os.makedirs(MAIN_OUTPUT_DIR)

def get_color_stats(img, mask):
    """计算图片在 mask 区域内的均值和标准差"""
    valid_pixels = img[mask > 128]
    if len(valid_pixels) == 0:
        return np.zeros(3), np.zeros(3)
    mean = np.mean(valid_pixels, axis=0)
    std = np.std(valid_pixels, axis=0)
    return mean, std

def apply_color_transfer(style_img, mean_target, std_target):
    """LAB 颜色迁移"""
    style_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2LAB).astype(float)
    mean_style = np.mean(style_img, axis=(0,1))
    std_style = np.std(style_img, axis=(0,1))
    std_style[std_style == 0] = 1e-5
    
    for i in range(3):
        style_img[:,:,i] = (style_img[:,:,i] - mean_style[i]) + mean_target[i]
        
    style_img = np.clip(style_img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(style_img, cv2.COLOR_LAB2BGR)

# ==========================================
# 核心任务函数：封装了原本的所有逻辑
# ==========================================
def run_style_transfer(style_filename):
    print(f"\n{'='*40}")
    print(f"正在处理风格图: {style_filename}")
    print(f"{'='*40}")

    # 1. 准备当前任务的输出目录
    style_name_no_ext = os.path.splitext(style_filename)[0] # 去掉后缀，如 s1
    current_output_dir = os.path.join(MAIN_OUTPUT_DIR, style_name_no_ext)
    
    if not os.path.exists(current_output_dir):
        os.makedirs(current_output_dir)

    style_path = os.path.join("data/style-images", style_filename)

    # ================= 预处理：生成双风格图 =================
    content = cv2.imread(CONTENT_PATH)
    style = cv2.imread(style_path)
    mask = cv2.imread(MASK_PATH, 0)

    if content is None or style is None or mask is None:
        print(f"❌ 错误：无法读取文件 {style_filename}，跳过...")
        return

    # 缩放 mask
    mask = cv2.resize(mask, (content.shape[1], content.shape[0]), interpolation=cv2.INTER_NEAREST)
    content_lab = cv2.cvtColor(content, cv2.COLOR_BGR2LAB).astype(float)

    # 统计信息
    mean_bg, std_bg = get_color_stats(content_lab, mask) # mask>128 (白) -> 背景
    mask_fg_cv = cv2.bitwise_not(mask)
    mean_fg, std_fg = get_color_stats(content_lab, mask_fg_cv) # mask_fg>128 -> 前景

    # 生成并保存临时风格图 (存在各自的文件夹里)
    style_fg = apply_color_transfer(style.copy(), mean_fg, std_fg)
    style_fg_path = os.path.join(current_output_dir, "style_fg.jpg")
    cv2.imwrite(style_fg_path, style_fg)

    style_bg = apply_color_transfer(style.copy(), mean_bg, std_bg)
    style_bg_path = os.path.join(current_output_dir, "style_bg.jpg")
    cv2.imwrite(style_bg_path, style_bg)

    # ================= TensorFlow 图构建 =================
    # 重要：每次循环必须重置 TF 图，否则节点会累积导致显存爆炸
    tf.reset_default_graph() 

    # 权重配置 (保持你的参数)
    BG_LAPLACE_WEIGHT = 0.05 
    BG_CONTENT_WEIGHT = 0.1
    BG_STYLE_RATIO = 1
    FG_STYLE_RATIO = 0.2
    STYLE_LAYER_WEIGHTS = [20, 20, 20, 20, 5000]
    
    # 定义网络操作辅助函数
    def make_kernel(a):
        a = np.asarray(a)
        a = a.reshape(list(a.shape) + [1, 1])
        return tf.constant(a, dtype=1)

    def simple_conv(x, k):
        num_channels = int(x.get_shape()[-1])
        k_tiled = tf.tile(k, [1, 1, num_channels, 1])
        y = tf.nn.depthwise_conv2d(x, k_tiled, strides=[1, 1, 1, 1], padding='SAME')
        return y

    def laplace(x):
        laplace_k = make_kernel([[0.5, 1.0, 0.5], [1.0, -6., 1.0], [0.5, 1.0, 0.5]])
        return simple_conv(x, laplace_k)

    def compute_masked_gram(feature_map, mask_resized):
        masked_features = feature_map * mask_resized
        num_channels = tf.cast(tf.shape(feature_map)[3], tf.float32)
        valid_pixels = tf.reduce_sum(mask_resized) + 1e-5
        flat = tf.reshape(masked_features, [-1, tf.shape(feature_map)[3]])
        gram = tf.matmul(flat, flat, transpose_a=True) / (2.0 * num_channels * valid_pixels)
        return gram

    def compute_global_gram(feature_map):
        num_channels = tf.cast(tf.shape(feature_map)[3], tf.float32)
        num_pixels = tf.cast(tf.reduce_prod(tf.shape(feature_map)[1:3]), tf.float32)
        flat = tf.reshape(feature_map, [-1, tf.shape(feature_map)[3]])
        gram = tf.matmul(flat, flat, transpose_a=True) / (2.0 * num_channels * num_pixels)
        return gram

    # 加载数据
    model = vgg.Vgg19()
    content_img_np = np.asarray(PIL.Image.open(CONTENT_PATH).convert('RGB'), dtype=float)
    img_width = content_img_np.shape[0]
    img_height = content_img_np.shape[1]

    style_fg_np = np.asarray(PIL.Image.open(style_fg_path).convert('RGB'))
    style_fg_resized = tf.image.resize_images(style_fg_np, size=[img_width, img_height])

    style_bg_np = np.asarray(PIL.Image.open(style_bg_path).convert('RGB'))
    style_bg_resized = tf.image.resize_images(style_bg_np, size=[img_width, img_height])

    mask_np_src = np.asarray(PIL.Image.open(MASK_PATH).convert('L').resize((img_height, img_width)), dtype=float) / 255.0
    mask_tensor = tf.reshape(tf.constant(mask_np_src, dtype=tf.float32), [1, img_width, img_height, 1])
    
    # Mask定义
    mask_bg = mask_tensor 
    mask_fg = 1 - mask_tensor

    # 混合初始化逻辑
    b = np.zeros(shape=[1, img_width, img_height, 3])
    b[0] = content_img_np
    
    noise_np = np.random.normal(loc=128.0, scale=30.0, size=(1, img_width, img_height, 3)).astype(np.float32)
    noise_np = np.clip(noise_np, 0, 255)
    
    mask_broadcast = mask_np_src.reshape(1, img_width, img_height, 1)
    
    # 混合初始化：Mask=0(FG)用原图，Mask=1(BG)用噪声
    hybrid_init_np = b * (1.0 - mask_broadcast) + noise_np * mask_broadcast
    
    # 变量
    raw_image_var = tf.Variable(hybrid_init_np, trainable=True, dtype=tf.float32)
    input_var = tf.clip_by_value(raw_image_var, 0.0, 255.0)

    model.build(input_var)
    
    # 会话配置
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    # ================= 训练过程 =================
    with tf.Session(config=config) as sess:
        layers = [model.conv1_1, model.conv2_1, model.conv3_1, model.conv4_1, model.conv5_1]

        # 计算 Targets
        target_grams_fg = []
        style_fg_val = sess.run(style_fg_resized)
        sess.run(raw_image_var.assign(style_fg_val.reshape(1, img_width, img_height, 3)))
        for layer in layers:
            target_grams_fg.append(sess.run(compute_global_gram(layer)))

        target_grams_bg = []
        style_bg_val = sess.run(style_bg_resized)
        sess.run(raw_image_var.assign(style_bg_val.reshape(1, img_width, img_height, 3)))
        for layer in layers:
            target_grams_bg.append(sess.run(compute_global_gram(layer)))
        
        # 恢复Content计算Content Loss Target
        sess.run(raw_image_var.assign(b))
        
        # --- Loss 构建 ---
        # 1. Style Loss (分层计算)
        style_losses = []
        for i, layer in enumerate(layers):
            h, w = int(layer.shape[1]), int(layer.shape[2])
            m_bg_l = tf.image.resize_nearest_neighbor(mask_bg, [h, w])
            m_fg_l = tf.image.resize_nearest_neighbor(mask_fg, [h, w])
            
            # FG
            gram_fg = compute_masked_gram(layer, m_fg_l)
            loss_fg = tf.reduce_sum(tf.square(gram_fg - target_grams_fg[i]))
            
            # BG
            gram_bg = compute_masked_gram(layer, m_bg_l)
            loss_bg = tf.reduce_sum(tf.square(gram_bg - target_grams_bg[i]))
            
            layer_loss = (FG_STYLE_RATIO * loss_fg + BG_STYLE_RATIO * loss_bg) * STYLE_LAYER_WEIGHTS[i]
            style_losses.append(layer_loss)
            
        style_loss = sum(style_losses) / 5.0

        # 2. Content Loss
        def weighted_content_loss(layer, target):
            h, w = int(layer.shape[1]), int(layer.shape[2])
            sw = tf.image.resize_nearest_neighbor(mask_fg, [h, w])*1.0 + \
                 tf.image.resize_nearest_neighbor(mask_bg, [h, w])*BG_CONTENT_WEIGHT
            diff = tf.square(layer - target) * sw
            return tf.reduce_sum(diff) / (2.0 * np.sqrt(h * w * int(layer.shape[3])))

        t_c42 = sess.run(model.conv4_2, feed_dict={input_var: b})
        t_c12 = sess.run(model.conv1_2, feed_dict={input_var: b})
        content_loss = (weighted_content_loss(model.conv4_2, t_c42) + 
                        weighted_content_loss(model.conv1_2, t_c12)) / 2.0

        # 3. Laplace Loss
        laplace_loss = 0.0
        lap_spatial_weight = mask_fg * 1.0 + mask_bg * BG_LAPLACE_WEIGHT 
        for p_size in [1, 2, 4, 8, 10, 16]:
            if p_size == 1:
                p_in = input_var
            else:
                p_in = tf.nn.pool(input_var, window_shape=[p_size, p_size], pooling_type='AVG', padding='SAME', strides=[p_size, p_size])
                p_in = tf.nn.pool(input_var, window_shape=[p_size, p_size], pooling_type='AVG', padding='SAME', strides=[p_size, p_size])
            
            t_shape = tf.shape(p_in)[1:3]
            c_weight = tf.image.resize_nearest_neighbor(lap_spatial_weight, t_shape)
            
            lap_op = laplace(p_in)
            t_lap = sess.run(lap_op, feed_dict={raw_image_var: b})
            
            weighted_diff = tf.square(lap_op - t_lap) * c_weight
            laplace_loss += tf.reduce_mean(weighted_diff)

        # 4. Total Loss & Opt
        coefs = [1e7, 1e5, 1e10] # Style, Content, Laplace
        total_loss = coefs[0]*style_loss + coefs[1]*content_loss + coefs[2]*laplace_loss
        
        train_op = tf.contrib.opt.ScipyOptimizerInterface(total_loss, method='L-BFGS-B', options={'maxiter': 1000})
        
        # 初始化
        sess.run(tf.global_variables_initializer())
        print(f"[{style_name_no_ext}] 应用混合初始化...")
        sess.run(raw_image_var.assign(hybrid_init_np))

        # 回调函数
        # 为了不让打印太乱，我们定义一个简单的计数器类
        class StepCounter:
            def __init__(self):
                self.step = 0
        
        counter = StepCounter()

        def callback(tl, cl, sl, ll, ii):
            if counter.step % 100 == 0:
                print(f"[{style_name_no_ext}] Iter: {counter.step:4d} | Total: {tl:.2e}")
                # 保存中间结果到对应子文件夹
                img_save = PIL.Image.fromarray(tf.cast(ii, dtype=tf.uint8).eval(session=sess)[0], 'RGB')
                save_name = os.path.join(current_output_dir, f"iter_{counter.step}.png")
                img_save.save(save_name)
            counter.step += 1

        train_op.minimize(sess, 
                          fetches=[total_loss, content_loss, style_loss, laplace_loss, input_var], 
                          loss_callback=callback)

        # 保存最终结果
        final_img = PIL.Image.fromarray(input_var.eval(session=sess)[0], 'RGB')
        final_save_path = os.path.join(current_output_dir, "final_result.png")
        final_img.save(final_save_path)
        print(f"✅ [{style_name_no_ext}] 完成！结果已保存至 {final_save_path}")

# ==========================================
# 主程序入口：循环执行
# ==========================================
if __name__ == "__main__":
    print(f"开始批量处理，共 {len(STYLE_IMAGES_LIST)} 张风格图...")
    
    for style_file in STYLE_IMAGES_LIST:
        run_style_transfer(style_file)
        
    print("\n🎉 所有任务全部完成！请检查 data/batch_results_dual 文件夹。")