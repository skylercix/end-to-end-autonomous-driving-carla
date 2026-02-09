import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# --- 1. Definim Arhitectura (Exact ca la antrenare) ---
class SmallNvidiaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Index 0: Conv1
            nn.Conv2d(3, 24, 5, stride=2), nn.ReLU(),
            # Index 2: Conv2
            nn.Conv2d(24, 36, 5, stride=2), nn.ReLU(),
            # Index 4: Conv3
            nn.Conv2d(36, 48, 5, stride=2), nn.ReLU(),
            # Index 6: Conv4
            nn.Conv2d(48, 64, 3), nn.ReLU(),
            # Index 8: Conv5
            nn.Conv2d(64, 64, 3), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 1 * 18, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 10), nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.net(x)

# --- 2. Configurare ---
MODEL_PATH = "model.pth"
DATASET_DIR = "dataset_small"
DEVICE = "cpu"

# Transformarile (trebuie sa fie identice cu cele de la antrenare)
transform = transforms.Compose([
    transforms.Resize((66, 200)),
    transforms.ToTensor(),
])

# --- 3. Functia de Hook ---
# Aceasta lista va stoca iesirea stratului pe care il monitorizam
activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def main():
    # A. Incarcam Modelul
    if not os.path.exists(MODEL_PATH):
        print("Nu am gasit model.pth!")
        return

    model = SmallNvidiaModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # B. Alegem o imagine random din dataset
    all_episodes = [os.path.join(DATASET_DIR, d) for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    if not all_episodes:
        print("Nu am gasit episoade in dataset!")
        return
    
    random_episode = random.choice(all_episodes)
    # Filtram doar fisierele png
    images = [f for f in os.listdir(random_episode) if f.endswith('.png')]
    
    if not images:
        print("Nu am gasit imagini in episod!")
        return

    image_path = os.path.join(random_episode, random.choice(images))
    print(f"Analizam imaginea: {image_path}")

    # C. Procesam imaginea
    raw_image = Image.open(image_path).convert("RGB")
    input_tensor = transform(raw_image).unsqueeze(0).to(DEVICE) # Adaugam batch dimension [1, 3, 66, 200]

    # D. Atasam Hook-ul
    # Vrem sa vedem ce scoate PRIMUL strat convolutional (Conv2d 3->24)
    # In nn.Sequential, acesta este la indexul 0.
    # Daca vrei sa vezi straturi mai adanci, schimba indexul (ex: model.net[2] pentru al doilea conv)
    
    layer_to_visualize = model.net[0] 
    layer_to_visualize.register_forward_hook(get_activation('conv1'))

    # E. Trecem imaginea prin model
    _ = model(input_tensor)

    # F. Extragem Feature Map-urile
    act = activation['conv1'].squeeze() # Eliminam dimensiunea batch-ului -> [24, H, W]
    
    # G. Vizualizare cu Matplotlib
    num_filters = act.shape[0] # Ar trebui sa fie 24
    
    fig = plt.figure(figsize=(15, 8))
    
    # Afisam imaginea originala
    plt.subplot(5, 6, 1)
    plt.imshow(raw_image)
    plt.title("Original")
    plt.axis('off')

    # Afisam fiecare filtru
    for i in range(num_filters):
        ax = plt.subplot(5, 6, i + 7) # +7 pentru ca incepem dupa original si cateva spatii goale
        feature_map = act[i].cpu().numpy()
        
        # Folosim 'viridis' sau 'gray' pentru a vedea activarile
        ax.imshow(feature_map, cmap='viridis') 
        ax.axis('off')
        ax.set_title(f'Filter {i}')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()