import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import torch.nn.functional as F


MODEL_PATH = "model_nav_traffic.pth"
DATASET_DIR = "dataset_traffic_processed"
DEVICE = "cpu"  


class ConditionalNvidiaModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, 5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, 5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, 3), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU(),
            nn.Flatten()
        )
        
        self.command_fc = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU()
        )

        self.tl_fc = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU()
        )

        self.joint_fc = nn.Sequential(
            nn.Linear(1152 + 16 + 16, 256), nn.ReLU(),
            nn.Dropout(p=0.3), 
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(p=0.2), 
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 3) 
        )

    def forward(self, img, cmd, tl):
        img_features = self.conv_layers(img)
        cmd_onehot = F.one_hot(cmd.long(), num_classes=4).float()
        cmd_features = self.command_fc(cmd_onehot)
        tl_onehot = F.one_hot(tl.long(), num_classes=3).float()
        tl_features = self.tl_fc(tl_onehot)
        combined = torch.cat((img_features, cmd_features, tl_features), dim=1)
        return self.joint_fc(combined)


def convert_yuv(img):
    return img.convert("YCbCr")

transform = transforms.Compose([
    transforms.Lambda(convert_yuv),     
    transforms.ToTensor(),              
])


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def main():
    if not os.path.exists(MODEL_PATH):
        print(f" Eroare: nu exista {MODEL_PATH}")
        return
    if not os.path.exists(DATASET_DIR):
        print(f"Eroare: nu exista {DATASET_DIR}")
        return

    print("Se incarca modelul...")
    model = ConditionalNvidiaModel().to(DEVICE)
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

    raw_image = Image.open(image_path).convert("RGB")
    

    input_tensor = transform(raw_image).unsqueeze(0).to(DEVICE)
    
  
    cmd_tensor = torch.tensor([3], dtype=torch.long).to(DEVICE)
    tl_tensor = torch.tensor([0], dtype=torch.long).to(DEVICE)

   
    layer_to_visualize = model.conv_layers[0] 
    layer_to_visualize.register_forward_hook(get_activation('conv1'))

    _ = model(input_tensor, cmd_tensor, tl_tensor)

 
    act = activation['conv1'].squeeze()
    num_filters = act.shape[0] 
    
    print(f"Vizualizam {num_filters} filtre din primul strat convolutional.")

   
    fig = plt.figure(figsize=(15, 8))
    
    input_img_display = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    
    plt.subplot(5, 6, 1)
    plt.imshow(input_img_display)
    plt.title("Input (YUV)", fontsize=10)
    plt.axis('off')

    for i in range(num_filters):
        ax = plt.subplot(5, 6, i + 2) 
        feature_map = act[i].cpu().numpy()
        
        ax.imshow(feature_map, cmap='viridis') 
        ax.axis('off')
        ax.set_title(f'Filtru {i}', fontsize=8)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()