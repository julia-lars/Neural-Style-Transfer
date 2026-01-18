import torch
import torchvision
from torch import nn
import os
import matplotlib.pyplot as plt
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image
import numpy as np
from scipy.stats import wasserstein_distance

# 1. 设备配置 ---
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}", flush=True)

# 2. 图片加载与分辨率控制 ---
content_path = "data/content-images/c2.png"
style_path   = "data/style-images/s2.jpg"

content_img_pil = Image.open(content_path)
style_img_pil   = Image.open(style_path)

rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std  = torch.tensor([0.229, 0.224, 0.225])

def gatys_preprocess_from_pil(img):
    # size_hw: (H, W) 或 int
    #img = torchvision.transforms.Resize(size_hw)(img)
    x = torchvision.transforms.ToTensor()(img).unsqueeze(0).to(device)
    x = x * 255.0
    x = x[:, [2,1,0], :, :]  # RGB -> BGR
    mean = torch.tensor([103.939, 116.779, 123.68]).view(1,3,1,1).to(device)
    return x - mean

def vgg_normalize(tensor):
    mean = rgb_mean.view(1, 3, 1, 1).to(device)
    std = rgb_std.view(1, 3, 1, 1).to(device)
    return (tensor - mean) / std

# 4. 模型构建 ---
vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:29]
for i, layer in enumerate(vgg):
    if isinstance(layer, nn.MaxPool2d):
        vgg[i] = nn.AvgPool2d(kernel_size=2, stride=2) #将最大池化层换位平均池化层
vgg = vgg.to(device).eval()
for p in vgg.parameters():
    p.requires_grad = False

# 5. 特征提取配置 ---
style_layers  = [0, 5, 10, 19, 28]  
content_layer = [21]               

# 保持你修改后的权重：侧重深层
style_layer_weights_stage1 = [1e3/n**2 for n in [64,128,256,512,512]]
style_layer_weights_stage2 = [10e3/n**2 for n in [64,128,256,512,512]]
def extract_features(x):
    contents, styles = [], []
    for i, layer in enumerate(vgg):
        x = layer(x)
        if i in content_layer:
            contents.append(x)
        if i in style_layers:
            styles.append(x)
    return contents, styles

# 6. Loss 函数 ---
def content_loss(y_hat, y): #内容损失
    return torch.mean((y_hat - y.detach()) ** 2)

def gram(x):
    b, c, h, w = x.shape
    x = x.view(b, c, h*w)
    G = torch.bmm(x, x.transpose(1, 2)) / (h * w)
    return G

def style_loss(y_hat, g):
    return torch.mean((gram(y_hat) - g.detach()) ** 2)

def total_variation_loss(x): 
    return torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
           torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))

content_raw = gatys_preprocess_from_pil(content_img_pil)
style_raw   = gatys_preprocess_from_pil(style_img_pil)

with torch.no_grad():
    content_targets, _ = extract_features(content_raw)
    _, style_targets   = extract_features(style_raw)
    style_grams = [gram(y) for y in style_targets]

# 8. 训练设置
gen_img = nn.Parameter(content_raw.clone()) 
#gen_img = nn.Parameter(torch.randn(content_raw.shape).to(device) * 1e-1) #初始化生成图像

tv_weight      = 0

optimizer = torch.optim.LBFGS([gen_img], lr=1.0, max_iter=20, history_size=10, line_search_fn="strong_wolfe") # 可选，增强收敛

# 外层循环次数
num_steps = 50

os.makedirs("progress_images", exist_ok=True)
history = {'loss': [], 'style': [], 'content': []}
dist_history = []
dist_steps = []

print("开始优化 (L-BFGS)... ", flush=True)

# --- 9. 训练循环 (L-BFGS 写法) ---
step_counter = 0
alpha = 1.0
beta = 1e3
def closure():
    optimizer.zero_grad()

    contents_hat, styles_hat = extract_features(gen_img)

    style_loss_total = sum(
        w * style_loss(y_hat, g)
        for y_hat, g, w in zip(styles_hat, style_grams, style_layer_weights_stage1)
    )

    content_loss_total = content_loss(contents_hat[0], content_targets[0])
    total_variation = total_variation_loss(gen_img)
    loss = style_loss_total + content_loss_total + tv_weight * total_variation
    loss.backward()
    return loss
def gatys_postprocess(x):
    """
    x: BGR, mean-subtracted, [*, 3, H, W]
    return: RGB, [0,1]
    """
    x = x.clone()
    mean = torch.tensor([103.939, 116.779, 123.68]).view(1,3,1,1).to(x.device)
    x = x + mean
    x = x[:, [2,1,0], :, :]  # BGR → RGB
    x = x / 255.0
    return torch.clamp(x, 0.0, 1.0)
    
for i in range(num_steps):
    optimizer.step(closure)
    
with torch.no_grad():
    img_to_save = gatys_postprocess(gen_img.detach())
    img_pil = torchvision.transforms.ToPILImage()(img_to_save.cpu()[0])
    img_pil.save("gatys.png")

