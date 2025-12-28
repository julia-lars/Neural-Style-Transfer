import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}", flush=True)

content_img = Image.open("data/content-images/c1.jpg")
style_img   = Image.open("data/style-images/s1.jpg")

h, w = content_img.size[1], content_img.size[0]
image_shape = (h, w)

# VGG preprocessing
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std  = torch.tensor([0.229, 0.224, 0.225])

def preprocess(img):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0).to(device)

def vgg_normalize(tensor):
    """Applies VGG mean/std normalization."""
    mean = rgb_mean.view(1, 3, 1, 1).to(device)
    std = rgb_std.view(1, 3, 1, 1).to(device)
    return (tensor - mean) / std

def postprocess(x):
    x = x.detach().cpu()[0]
    x = x * rgb_std.view(3, 1, 1) + rgb_mean.view(3, 1, 1)
    x = torch.clamp(x,0,1)
    return torchvision.transforms.ToPILImage()(x)

#pretrained VGG NN (only convolutional layers)
vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:29]
for i, layer in enumerate(vgg):
    if isinstance(layer, nn.MaxPool2d):
        vgg[i] = nn.AvgPool2d(kernel_size=2, stride=2)
vgg = vgg.to(device).eval()
for p in vgg.parameters():
    p.requires_grad = False

# Layers (paper-correct)
style_layers   = [0, 5, 10, 19, 28]  # conv1_1 ... conv5_1
content_layer = [21]               # conv4_2

# Feature extraction
def extract_features(x):
    contents, styles = [], []
    for i, layer in enumerate(vgg):
        x = layer(x)
        if i in content_layer: #save the activation layer(tensor) for content loss
            contents.append(x)
        if i in style_layers: #save the activation layer(tensor) for style loss
            styles.append(x)
    return contents, styles

# Losses
def content_loss(y_hat, y): #(feature map of generated image(at a content lyer), feature map of content image)
    return torch.mean((y_hat - y.detach()) ** 2)

def gram(x): #compute the gram matrix
    c, h, w = x.shape[1:] #feature channels, h x w
    x = x.view(c, h * w)
    x = torch.clamp(x, -1e3, 1e3)
    return (x @ x.T) / (c * h * w)

def style_loss(y_hat, g): #(generated image features(at feature layer), gram matrix of style image features)
    return torch.mean((gram(y_hat) - g.detach()) ** 2)

def total_variation_loss(x):
    return torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
           torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))

# Prepare targets
# 1. Prepare Content/Style in RAW pixel space (0 to 1)
content_raw = preprocess(content_img) # Just Resize and ToTensor
style_raw   = preprocess(style_img)

# 2. Extract targets using a temporary normalized version
with torch.no_grad():
    content_targets, _ = extract_features(vgg_normalize(content_raw))
    _, style_targets   = extract_features(vgg_normalize(style_raw))
    style_grams = [gram(y) for y in style_targets]

# 3. Initialize Parameter in RAW pixel space
gen_img = nn.Parameter(content_raw.clone())

content_weight = 1.0
style_weight   = 1e4   
tv_weight      = 1e-2

# 4. Optimizer: Use lr=1.0 for LBFGS (standard for line search)
optimizer = torch.optim.LBFGS([gen_img], lr=1.0)
num_steps = 30
for i in range(num_steps):
    def closure():
        optimizer.zero_grad()
        
        # Normalize ONLY for the VGG pass
        normalized_gen = vgg_normalize(gen_img)
        contents_hat, styles_hat = extract_features(normalized_gen)

        c_loss = sum(content_loss(a, b) for a, b in zip(contents_hat, content_targets))
        s_loss = sum(style_loss(a, b) for a, b in zip(styles_hat, style_grams))
        tv_loss = total_variation_loss(gen_img)

        loss = (content_weight * c_loss) + (style_weight * s_loss) + (tv_weight * tv_loss)
        loss.backward()
        return loss

    optimizer.step(closure)
    
    # Keep the actual image in the valid [0, 1] range
    with torch.no_grad():
        gen_img.clamp_(0, 1)

    if (i + 1) % 10 == 0:
        print(f"Step {i+1}/{num_steps} complete", flush=True)

# 5. Corrected Postprocess (since gen_img is already 0-1)
final_result = torchvision.transforms.ToPILImage()(gen_img.detach().cpu()[0])
final_result.save("stylized_final.png")