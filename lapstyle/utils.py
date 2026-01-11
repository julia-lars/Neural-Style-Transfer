import numpy as np
import cv2 # 需要安装 opencv-python，如果不允许装库，可以用 PIL 实现但比较麻烦
from PIL import Image

def transfer_color(src, dest):
    """
    将 dest(内容图) 的色调迁移到 src(风格图) 上。
    src: 风格图 (numpy array)
    dest: 内容图 (numpy array)
    返回: 变色后的风格图
    """
    # 1. 转换到 LAB 颜色空间 (L=亮度, A/B=颜色通道)
    src = cv2.cvtColor(src, cv2.COLOR_RGB2LAB)
    dest = cv2.cvtColor(dest, cv2.COLOR_RGB2LAB)
    
    # 2. 分离通道
    src_l, src_a, src_b = cv2.split(src)
    dest_l, dest_a, dest_b = cv2.split(dest)
    
    # 3. 计算均值和标准差
    src_stats = [s.mean() for s in [src_l, src_a, src_b]]
    src_std = [s.std() for s in [src_l, src_a, src_b]]
    dest_stats = [d.mean() for d in [dest_l, dest_a, dest_b]]
    dest_std = [d.std() for d in [dest_l, dest_a, dest_b]]
    
    # 4. 颜色迁移公式: (x - mean_src) * (std_dest / std_src) + mean_dest
    # 注意：我们通常保留风格图的 L (亮度) 分量，或者也进行迁移。这里对三个通道都迁移效果最稳。
    src_l = ((src_l - src_stats[0]) * (dest_std[0] / (src_std[0] + 1e-5)) + dest_stats[0])
    src_a = ((src_a - src_stats[1]) * (dest_std[1] / (src_std[1] + 1e-5)) + dest_stats[1])
    src_b = ((src_b - src_stats[2]) * (dest_std[2] / (src_std[2] + 1e-5)) + dest_stats[2])
    
    # 5. 裁剪范围并合并
    src_l = np.clip(src_l, 0, 255).astype(np.uint8)
    src_a = np.clip(src_a, 0, 255).astype(np.uint8)
    src_b = np.clip(src_b, 0, 255).astype(np.uint8)
    
    result = cv2.merge([src_l, src_a, src_b])
    
    # 6. 转回 RGB
    result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
    return result

def prepare_style_image_with_content_color(content_path, style_path, output_path):
    # 读取图片
    content = np.asarray(Image.open(content_path).convert('RGB'))
    style = np.asarray(Image.open(style_path).convert('RGB'))
    
    # 调整 style 大小以匹配 content (为了计算方便，虽非必须但推荐)
    style = cv2.resize(style, (content.shape[1], content.shape[0]))
    
    # 迁移颜色
    new_style = transfer_color(style, content)
    
    # 保存
    Image.fromarray(new_style).save(output_path)
    print(f"生成的特定色调风格图已保存至: {output_path}")

import cv2
import numpy as np
from sklearn.cluster import KMeans

def generate_kmeans_mask(image_path, output_path):
    """
    使用 K-Means 自动将图片分为前景和后景
    """
    # 1. 读取图片
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_shape = img.shape
    
    # 2. 展平图片 (Height * Width, 3)
    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # 3. K-Means 聚类 (聚成 2 类：背景和前景)
    kmeans = KMeans(n_clusters=2, random_state=0)
    labels = kmeans.fit_predict(pixel_values)
    
    # 4. 恢复形状
    mask = labels.reshape((original_shape[0], original_shape[1]))
    
    # 5. 自动判断哪一个是天空/背景
    # 假设：天空通常在图片的上半部分。计算上半部分的 label 0 多还是 label 1 多。
    half_h = original_shape[0] // 3
    top_region = mask[:half_h, :]
    if np.sum(top_region == 0) > np.sum(top_region == 1):
        bg_label = 0
    else:
        bg_label = 1
        
    # 6. 生成 Mask (背景为白色 255，前景为黑色 0)
    final_mask = np.where(mask == bg_label, 255, 0).astype(np.uint8)
    
    # 可选：进行一点形态学操作，去除噪点
    kernel = np.ones((5,5), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    
    # 7. 保存
    cv2.imwrite(output_path, final_mask)
    print(f"K-Means Mask 已保存: {output_path}")

# 使用方法：
# generate_kmeans_mask("data/content-images/c2.jpg", "data/mask.jpg")

def generate_saliency_mask(image_path, output_path):
    """
    使用 OpenCV 的显著性检测自动寻找前景
    """
    img = cv2.imread(image_path)
    
    # 初始化显著性检测器
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliency.computeSaliency(img)
    
    if success:
        # 显著性图是 0-1 的浮点数，转换为 0-255
        saliencyMap = (saliencyMap * 255).astype("uint8")
        
        # 自动阈值分割 (Otsu's method)
        thresh, binary_mask = cv2.threshold(saliencyMap, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # 注意：显著性检测通常把“前景”标为白。
        # 如果你的逻辑是“白色代表天空(要强风格)”，你需要反转颜色
        inverted_mask = cv2.bitwise_not(binary_mask)
        
        cv2.imwrite(output_path, inverted_mask)
        print(f"Saliency Mask 已保存: {output_path}")

import os

def visualize_comparison(title, img_list, save_path):
    """
    辅助函数：将多张图片横向拼接并保存，方便对比
    img_list: 包含 numpy 数组的列表
    """
    # 确保所有图片高度一致（以第一张为准），方便拼接
    h_target = img_list[0].shape[0]
    resized_imgs = []
    
    for img in img_list:
        # 如果是灰度图(mask)，转为RGB以便拼接
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # 调整高度，保持宽高比
        h, w = img.shape[:2]
        scale = h_target / h
        new_w = int(w * scale)
        resized = cv2.resize(img, (new_w, h_target))
        resized_imgs.append(resized)
    
    # 横向拼接
    concat_img = np.concatenate(resized_imgs, axis=1)
    
    # 转换颜色空间 BGR -> RGB (因为 OpenCV 保存需要 BGR，但如果不转直接存也可以，视上一步来源而定)
    # 这里的 img_list 假设是 RGB (PIL读入) 或 BGR (OpenCV读入)。
    # 为了统一，建议最后保存前统一转为 BGR
    if title == "mask": # Mask 拼接通常不需要色彩转换
        cv2.imwrite(save_path, concat_img)
    else:
        # 如果输入是 RGB (PIL)，OpenCV 保存需要转 BGR
        save_img = cv2.cvtColor(concat_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, save_img)
    
    print(f"[{title}] 对比图已保存: {save_path}")

# ==========================================
#              主执行逻辑
# ==========================================
if __name__ == "__main__":
    # 1. 定义测试图片路径 (请确保这些文件存在)
    content_path = "data/content-images/c2.png"  # 猫
    style_path = "data/style-images/s2.jpg"      # 星空
    
    # 检查路径是否存在
    if not os.path.exists(content_path) or not os.path.exists(style_path):
        print("错误：找不到测试图片，请检查路径！")
        exit()

    # 创建输出文件夹
    output_dir = "debug_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("=== 开始测试色调迁移 ===")
    # 读取图片 (用于传入 transfer_color)
    content_img = np.asarray(Image.open(content_path).convert('RGB'))
    style_img = np.asarray(Image.open(style_path).convert('RGB'))
    
    # 调用你的函数
    # 注意：style_img 需要先 resize 到和 content_img 差不多大小，否则均值计算可能不准
    style_img_resized = cv2.resize(style_img, (content_img.shape[1], content_img.shape[0]))
    
    # 【测试函数 1】 transfer_color
    colored_style = transfer_color(style_img_resized, content_img)
    
    # 保存单张结果
    save_p = os.path.join(output_dir, "style_colored.jpg")
    Image.fromarray(colored_style).save(save_p)
    
    # 保存对比图 (风格原图 vs 变色后风格 vs 内容图参考)
    visualize_comparison("color_transfer", 
                         [style_img_resized, colored_style, content_img], 
                         os.path.join(output_dir, "debug_color_compare.jpg"))

    print("\n=== 开始测试 Mask 生成 ===")
    
    # 【测试函数 2】 K-Means Mask
    kmeans_out_path = os.path.join(output_dir, "mask_kmeans.jpg")
    generate_kmeans_mask(content_path, kmeans_out_path)
    
    # 读取刚生成的 mask 看看
    mask_k = cv2.imread(kmeans_out_path, 0) # 0 表示灰度读取
    
    # 【测试函数 3】 Saliency Mask (如果安装了 opencv-contrib-python)
    saliency_out_path = os.path.join(output_dir, "mask_saliency.jpg")
    try:
        generate_saliency_mask(content_path, saliency_out_path)
        mask_s = cv2.imread(saliency_out_path, 0)
        
        # 保存 Mask 对比图 (原图 vs KMeans vs Saliency)
        # 注意 content_img 是 RGB，cv2 读取的是 BGR，为了显示颜色正确，这里转换一下
        content_bgr = cv2.cvtColor(content_img, cv2.COLOR_RGB2BGR)
        visualize_comparison("mask", 
                             [content_bgr, mask_k, mask_s], 
                             os.path.join(output_dir, "debug_mask_compare.jpg"))
                             
    except AttributeError:
        print("提示：你的 OpenCV 版本不支持 Saliency 检测。")
        print("需要安装: pip install opencv-contrib-python")
        # 仅对比 KMeans
        content_bgr = cv2.cvtColor(content_img, cv2.COLOR_RGB2BGR)
        visualize_comparison("mask", 
                             [content_bgr, mask_k], 
                             os.path.join(output_dir, "debug_mask_compare.jpg"))

    print(f"\n所有结果已保存至文件夹: {output_dir}")