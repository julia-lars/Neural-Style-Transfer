import torch
import torchvision
from torch import nn
import os
import matplotlib.pyplot as plt
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image
import numpy as np
from scipy.stats import wasserstein_distance

def analyze_histogram(step, gen_img, style_img, layer_idx=4):
    """
    画出生成图和风格图在指定层的特征数值直方图，并计算统计距离。
    """
    with torch.no_grad():
        # 1. 提取特征
        _, gen_styles = extract_features(vgg_normalize(gen_img))
        _, style_targets = extract_features(vgg_normalize(style_img))
        
        # 2. 获取指定层的特征图并展平
        gen_data = gen_styles[layer_idx].detach().cpu().numpy().flatten()
        style_data = style_targets[layer_idx].detach().cpu().numpy().flatten()
        
    # 3. 计算统计距离
    dist = wasserstein_distance(gen_data, style_data)
    
    # 4. 画图
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
    
    # 保存
    save_path = f"progress_images/hist_step_{step}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"--> Histogram analysis saved. Distance: {dist:.4f}")
    
    return dist

def visualize_compare(step, gen_img, style_img, layer_idx=4):
    """
    可视化对比生成图和风格图在某一层出的特征。
    """
    # 1. 准备数据
    with torch.no_grad():
        _, gen_styles = extract_features(vgg_normalize(gen_img))
        _, style_targets = extract_features(vgg_normalize(style_img))
        
        gen_map = gen_styles[layer_idx].detach().cpu()
        style_map = style_targets[layer_idx].detach().cpu()
    
    # 2. 选取前 8 个通道 (Channel) 来画图
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
    print(f"--> Feature visualization saved for step {step}_layer_{layer_idx}")

# --- 1. 设备配置 ---
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}", flush=True)

# --- 2. 图片加载与分辨率控制 ---
content_path = "data/content-images/c1.jpg"
style_path   = "data/style-images/s2.jpg"

content_img_pil = Image.open(content_path)
style_img_pil   = Image.open(style_path)

max_size = 400 
w_orig, h_orig = content_img_pil.size
scale = max_size / max(w_orig, h_orig)
h, w = int(h_orig * scale), int(w_orig * scale)
image_shape = (h, w) 
print(f"Training resolution set to: {image_shape} (Height, Width)", flush=True)

# --- 3. 预处理函数 ---
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std  = torch.tensor([0.229, 0.224, 0.225])

def preprocess(img):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape), 
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

style_layer_weights = [0.1, 0.1, 0.2, 0.3, 0.3] 

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
style_raw   = preprocess(style_img_pil)

with torch.no_grad():
    content_targets, _ = extract_features(vgg_normalize(content_raw))
    _, style_targets   = extract_features(vgg_normalize(style_raw))
    style_grams = [gram(y) for y in style_targets]

# --- 8. 训练设置 ---
gen_img = nn.Parameter(torch.randn(content_raw.shape).to(device))  # 随机噪声初始化

content_weight = 0
style_weight   = 1e8   
tv_weight      = 0

optimizer = torch.optim.Adam([gen_img], lr=0.01)
num_steps = 6000
show_every = 500
save_every = 1500

os.makedirs("progress_images", exist_ok=True)
history = {'loss': [], 'style': [], 'content': []}
dist_history = []
dist_steps = []

print("开始优化...", flush=True)

# --- 【新增】Step 0: 记录初始状态 ---
with torch.no_grad():
    print("Recording Step 0 (Initial State)...", flush=True)
    
    # 1. 保存初始图片 (纯噪声)
    save_path = f"progress_images/step_0.png"
    torchvision.transforms.ToPILImage()(gen_img.detach().cpu()[0]).save(save_path)
    
    # 2. 可视化初始特征图 (Layer 4)
    visualize_compare(0, gen_img, style_raw, layer_idx=4)
    
    # 3. 计算初始直方图和距离
    current_dist = analyze_histogram(0, gen_img, style_raw, layer_idx=4)
    
    # 4. 存入列表，这样画折线图时就会从 t=0 开始
    dist_history.append(current_dist)
    dist_steps.append(0)

# --- 9. 训练循环 ---
for i in range(num_steps):
    optimizer.zero_grad()
    
    normalized_gen = vgg_normalize(gen_img)
    contents_hat, styles_hat = extract_features(normalized_gen)

    c_loss = sum(content_loss(a, b) for a, b in zip(contents_hat, content_targets))
    
    s_loss = 0
    for y_hat, g_target, w_layer in zip(styles_hat, style_grams, style_layer_weights):
        layer_loss = style_loss(y_hat, g_target)
        s_loss += layer_loss * w_layer 

    tv_loss = total_variation_loss(gen_img)

    loss = (content_weight * c_loss) + (style_weight * s_loss) + (tv_weight * tv_loss)
    
    loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        gen_img.clamp_(0, 1)

    if (i + 1) % show_every == 0:
        print(f"Step [{i+1}/{num_steps}] Total: {loss.item():.2f} | Style: {s_loss.item()*style_weight:.2f} | Content: {c_loss.item()*content_weight:.2f}", flush=True)
        
    if (i + 1) % save_every == 0:
        save_path = f"progress_images/step_{i+1}.png"
        torchvision.transforms.ToPILImage()(gen_img.detach().cpu()[0]).save(save_path)
        
        # 记录中间状态的直方图和距离
        visualize_compare(i+1, gen_img, style_raw, layer_idx=0)
        visualize_compare(i+1, gen_img, style_raw, layer_idx=1)
        visualize_compare(i+1, gen_img, style_raw, layer_idx=2)
        visualize_compare(i+1, gen_img, style_raw, layer_idx=3)
        visualize_compare(i+1, gen_img, style_raw, layer_idx=4)
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

plt.title(f"Feature Distribution Distance over Time\n(Lower is better)", fontsize=14)
plt.xlabel("Training Steps")
plt.ylabel("Wasserstein Distance")
plt.grid(True, alpha=0.3)
plt.legend()

dist_curve_path = "progress_images/wasserstein_distance_curve.png"
plt.savefig(dist_curve_path)
plt.show()

print(f"Dist 变化曲线已保存至: {dist_curve_path}")