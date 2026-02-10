import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import random


MODEL_PATH = "model.pth"
DATASET_DIR = "dataset_manual"
DEVICE = "cpu"  

# --- MODELUL
class SmallNvidiaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.ReLU(), # Layer 0 (Conv1)
            nn.Conv2d(24, 36, 5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, 5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, 3), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 1 * 18, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 10), nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.net(x)


def crop_img(img):
    return img.crop((0, 80, 320, 240))

def convert_yuv(img):
    return img.convert("YCbCr")

transform = transforms.Compose([
    transforms.Lambda(crop_img),        # 1. CROP
    transforms.Lambda(convert_yuv),     # 2. YUV
    transforms.Resize((66, 200)),       # 3. Resize
    transforms.ToTensor(),              # 4. Tensor
])

# --- 4. EXTRAGERE ACTIVARI (Hook) ---
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def main():
    # Verificari initiale
    if not os.path.exists(MODEL_PATH):
        print(f" Eroare: nu exista {MODEL_PATH}")
        return
    if not os.path.exists(DATASET_DIR):
        print(f"Eroare: nu exista {DATASET_DIR}")
        return

    # Incarcare Model
    print("Se incarca modelul...")
    model = SmallNvidiaModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    
   
    all_images = []
    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                all_images.append(os.path.join(root, file))
    
    if not all_images:
        print("Nu am gasit nicio imagine in dataset!")
        return

    image_path = random.choice(all_images)
    print(f"Analizam imaginea: {image_path}")

    # --- Procesare Imagine ---
    raw_image = Image.open(image_path).convert("RGB")
    
    # Aplicam transformarile
    input_tensor = transform(raw_image).unsqueeze(0).to(DEVICE)

    
   
    layer_to_visualize = model.net[0] 
    layer_to_visualize.register_forward_hook(get_activation('conv1'))

    # Forward pass
    _ = model(input_tensor)

    # Extragem datele
    act = activation['conv1'].squeeze()
    num_filters = act.shape[0] 
    
    print(f"Vizualizam {num_filters} filtre din primul strat convolutional.")

    
    fig = plt.figure(figsize=(15, 8))
    
    # 1. Afisam imaginea originala (prelucrata)
    # Convertim tensorul (C, H, W) -> (H, W, C) pentru matplotlib
    input_img_display = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    
    plt.subplot(5, 6, 1)
    plt.imshow(input_img_display)
    plt.title("Input (YUV)\n(Culorile par false)", fontsize=10)
    plt.axis('off')

    # 2. Afisam cele 24 de Feature Maps
    for i in range(num_filters):
        # Calculam pozitia in grid (incepem de la 2, pt ca 1 e imaginea originala)
        ax = plt.subplot(5, 6, i + 2) # Ajustat grid-ul ca sa incapem
        feature_map = act[i].cpu().numpy()
        
        ax.imshow(feature_map, cmap='viridis') # 'viridis', 'plasma', 'gray'
        ax.axis('off')
        ax.set_title(f'Filtru {i}', fontsize=8)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()