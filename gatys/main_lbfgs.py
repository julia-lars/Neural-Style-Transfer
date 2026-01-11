import torch
import torchvision
from torch import nn
import os
import matplotlib.pyplot as plt
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image
import numpy as np
from scipy.stats import wasserstein_distance

# --- 辅助函数保持不变 ---
def analyze_histogram(step, gen_img, style_img, layer_idx=4):
    with torch.no_grad():
        _, gen_styles = extract_features(vgg_normalize(gen_img))
        _, style_targets = extract_features(vgg_normalize(style_img))
        gen_data = gen_styles[layer_idx].detach().cpu().numpy().flatten()
        style_data = style_targets[layer_idx].detach().cpu().numpy().flatten()
    dist = wasserstein_distance(gen_data, style_data)
    
    plt.figure(figsize=(10, 6))
    bins = 100
    if len(style_data) > 0:
        range_limit = (0, np.percentile(style_data, 99))
    else:
        range_limit = (0, 1)
    plt.hist(style_data, bins=bins, range=range_limit, alpha=0.5, color='blue', label='Style Image Features', density=True)
    plt.hist(gen_data, bins=bins, range=range_limit, alpha=0.5, color='orange', label='Generated Image Features', density=True)
    plt.title(f"Step {step} - Feature Distribution (Layer {layer_idx})\nWasserstein Dist: {dist:.4f}", fontsize=14)
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency (Density)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = f"progress_images/hist_step_{step}.png"
    plt.savefig(save_path)
    plt.close()
    return dist

def visualize_compare(step, gen_img, style_img, layer_idx=4):
    with torch.no_grad():
        _, gen_styles = extract_features(vgg_normalize(gen_img))
        _, style_targets = extract_features(vgg_normalize(style_img))
        gen_map = gen_styles[layer_idx].detach().cpu()
        style_map = style_targets[layer_idx].detach().cpu()
    num_channels = min(8, gen_map.shape[1]) 
    fig, axes = plt.subplots(2, num_channels, figsize=(20, 5))
    fig.suptitle(f"Step {step}: Feature Maps Comparison (Layer index {layer_idx})", fontsize=16)
    for i in range(num_channels):
        ax = axes[0, i]
        s_img = style_map[0, i, :, :]
        ax.imshow(s_img, cmap='viridis')
        ax.axis('off')
        if i == 0: ax.set_title("Style Image Features", fontsize=12, loc='left')
        ax = axes[1, i]
        g_img = gen_map[0, i, :, :]
        ax.imshow(g_img, cmap='viridis')
        ax.axis('off')
        if i == 0: ax.set_title("Generated Image Features", fontsize=12, loc='left')
    plt.tight_layout()
    plt.savefig(f"progress_images/features_step_{step}_layer_{layer_idx}.png")
    plt.close()

# --- 1. 设备配置 ---
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}", flush=True)

# --- 2. 图片加载与分辨率控制 ---
content_path = "data/content-images/c2.png"
style_path   = "data/style-images/s2.jpg"

content_img_pil = Image.open(content_path)
style_img_pil   = Image.open(style_path)

def get_safe_size(size_orig, max_size=400):
    scale = max_size / max(size_orig)
    h, w = int(size_orig[1] * scale), int(size_orig[0] * scale)
    h = (h // 32) * 32
    w = (w // 32) * 32
    return h, w

h, w = get_safe_size(content_img_pil.size, max_size=400) # CPU上建议保持400或更小
image_shape = (h, w)
print(f"Optimized resolution: {image_shape}")

# --- 3. 预处理函数 ---
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std  = torch.tensor([0.229, 0.224, 0.225])

def preprocess(img):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape), 
        torchvision.transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0).to(device)

def preprocess_style(img, size=512): 
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size),
        torchvision.transforms.CenterCrop(size), # 加上CenterCrop比较稳妥
        torchvision.transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0).to(device)

def vgg_normalize(tensor):
    mean = rgb_mean.view(1, 3, 1, 1).to(device)
    std = rgb_std.view(1, 3, 1, 1).to(device)
    return (tensor - mean) / std

# --- 4. 模型构建 ---
vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:29]
for i, layer in enumerate(vgg):
    if isinstance(layer, nn.MaxPool2d):
        vgg[i] = nn.AvgPool2d(kernel_size=2, stride=2)
vgg = vgg.to(device).eval()
for p in vgg.parameters():
    p.requires_grad = False

# --- 5. 特征提取配置 ---
style_layers  = [0, 5, 10, 19, 28]  
content_layer = [21]               

# 保持你修改后的权重：侧重深层
style_layer_weights = [0.0, 0.0, 0.2, 0.4, 0.4] 

def extract_features(x):
    contents, styles = [], []
    for i, layer in enumerate(vgg):
        x = layer(x)
        if i in content_layer:
            contents.append(x)
        if i in style_layers:
            styles.append(x)
    return contents, styles

# --- 6. Loss 函数 ---
def content_loss(y_hat, y):
    return torch.mean((y_hat - y.detach()) ** 2)

def gram(x):
    c, h_feat, w_feat = x.shape[1:]
    x = x.view(c, h_feat * w_feat)
    return (x @ x.T) / (c * h_feat * w_feat)

def style_loss(y_hat, g):
    return torch.mean((gram(y_hat) - g.detach()) ** 2)

def total_variation_loss(x):
    return torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
           torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))

# --- 7. 准备目标数据 ---
content_raw = preprocess(content_img_pil)
# 使用较大的风格图尺寸
style_raw = preprocess_style(style_img_pil, size=512) 

with torch.no_grad():
    content_targets, _ = extract_features(vgg_normalize(content_raw))
    _, style_targets   = extract_features(vgg_normalize(style_raw))
    style_grams = [gram(y) for y in style_targets]

# --- 8. 训练设置 (关键修改) ---
# 【修改1】LBFGS 强烈建议使用 Content Image 初始化
#gen_img = nn.Parameter(content_raw.clone()) 
gen_img = nn.Parameter(torch.randn(content_raw.shape).to(device) * 1e-1)
# gen_img = nn.Parameter(torch.randn(content_raw.shape).to(device)) # 只有 Adam 可以试噪声

content_weight = 1000
style_weight   = 1e9
tv_weight      = 5

# 【修改3】使用 LBFGS 优化器
# max_iter=20: 每次 step 内部最多迭代 20 次 (CPU 友好设置)
optimizer = torch.optim.LBFGS([gen_img], 
                              lr=1.0, 
                              max_iter=20, 
                              history_size=10,
                              line_search_fn="strong_wolfe") # 可选，增强收敛

# 外层循环次数
num_steps = 50

os.makedirs("progress_images", exist_ok=True)
history = {'loss': [], 'style': [], 'content': []}
dist_history = []
dist_steps = []

print("开始优化 (L-BFGS)... 注意：每一步可能会比较慢，请耐心等待。", flush=True)

# --- Step 0: 记录初始状态 ---
with torch.no_grad():
    save_path = f"progress_images/step_0.png"
    torchvision.transforms.ToPILImage()(gen_img.detach().cpu()[0]).save(save_path)
    current_dist = analyze_histogram(0, gen_img, style_raw, layer_idx=4)
    dist_history.append(current_dist)
    dist_steps.append(0)

# --- 9. 训练循环 (L-BFGS 写法) ---
step_counter = 0

for i in range(num_steps):
    
    # 定义闭包函数 (L-BFGS 必须)
    def closure():
        global step_counter
        optimizer.zero_grad()
        
        normalized_gen = vgg_normalize(gen_img)
        contents_hat, styles_hat = extract_features(normalized_gen)

        c_loss = sum(content_loss(a, b) for a, b in zip(contents_hat, content_targets))
        
        s_loss = 0
        for y_hat, g_target, w_layer in zip(styles_hat, style_grams, style_layer_weights):
            s_loss += style_loss(y_hat, g_target) * w_layer 

        tv_loss = total_variation_loss(gen_img)

        loss = (content_weight * c_loss) + (style_weight * s_loss) + (tv_weight * tv_loss)
        
        loss.backward()
        
        # 打印部分放在 closure 里可以看到内部进度
        step_counter += 1
        if step_counter % 5 == 0:
            print(f"L-BFGS Eval {step_counter}: Total: {loss.item():.2f} | Style: {s_loss.item()*style_weight:.1f}", flush=True)
            
        return loss

    # 执行一步优化 (包含多次 eval)
    optimizer.step(closure)
    
    # 限制像素范围
    with torch.no_grad():
        gen_img.clamp_(0, 1)

    # 保存中间结果 (每一步外层循环都保存，因为 LBFGS 一步顶 Adam 几十步)
    print(f"--> 完成第 {i+1}/{num_steps} 次外层循环", flush=True)
    save_path = f"progress_images/step_lbfgs_{i+1}.png"
    torchvision.transforms.ToPILImage()(gen_img.detach().cpu()[0]).save(save_path)
    
    # 可视化分析
    if (i + 1) % 5 == 0:
        current_dist = analyze_histogram(i+1, gen_img, style_raw, layer_idx=4)
        dist_history.append(current_dist)
        dist_steps.append(i + 1)

# 保存最终图
final_result = torchvision.transforms.ToPILImage()(gen_img.detach().cpu()[0])
final_result.save("stylized_final_improved.png")
print("Done!")

# 画折线图
plt.figure(figsize=(10, 6))
plt.plot(dist_steps, dist_history, marker='o', linestyle='-', color='purple', label='Wasserstein Distance (Layer 4)')
plt.title(f"Feature Distribution Distance over Time", fontsize=14)
plt.savefig("progress_images/wasserstein_distance_curve.png")