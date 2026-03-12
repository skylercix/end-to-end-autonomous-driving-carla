import os
import csv
import random
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF


DATASET_DIR = "dataset_processed"
MODEL_SAVE_PATH = "model_nav.pth"

BATCH_SIZE = 64 
NUM_EPOCHS = 35
LEARNING_RATE = 1e-4    

NUM_WORKERS = 8        
PREFETCH_FACTOR = 2


if torch.cuda.is_available():
    DEVICE = "cuda"
    PIN_MEMORY = True   
    PERSISTENT_WORKERS = True if NUM_WORKERS > 0 else False
else:
    DEVICE = "cpu"
    PIN_MEMORY = False
    PERSISTENT_WORKERS = False


def convert_yuv(img):
    return img.convert("YCbCr")


class CarlaNavDataset(Dataset):
    def __init__(self, root):
        self.images = []
        self.steering = []
        self.commands = []

        if not os.path.exists(root):
             raise FileNotFoundError(f"Folderul {root} nu exista!")

        for episode_folder in os.listdir(root):
            episode_path = os.path.join(root, episode_folder)
            if not os.path.isdir(episode_path):
                continue
            
            csv_path = os.path.join(episode_path, "controls_nav.csv")
            if not os.path.exists(csv_path):
                continue
            
            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                try:
                    next(reader) 
                except StopIteration:
                    continue 

                for row in reader:
                    try:
                        img_name = row[0]
                        steer_val = float(row[1])
                        cmd_val = int(row[4]) 
                        
                        full_img_path = os.path.join(episode_path, img_name)
                        
                        if os.path.exists(full_img_path):
                            self.images.append(full_img_path)
                            self.steering.append(steer_val)
                            self.commands.append(cmd_val)
                    except (ValueError, IndexError):
                        continue 

        self.transform_pipeline = transforms.Compose([
            transforms.Lambda(convert_yuv),     
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.images[idx])
            img = self.transform_pipeline(img)
            
            steer = self.steering[idx]
            cmd = self.commands[idx]
            
            return img, torch.tensor(cmd, dtype=torch.float32), torch.tensor(steer, dtype=torch.float32)
            
        except Exception as e:
            return torch.zeros((3, 66, 200)), torch.tensor(0, dtype=torch.float32), torch.tensor(0.0, dtype=torch.float32)


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
            nn.Linear(1, 16), nn.ReLU()
        )

        self.joint_fc = nn.Sequential(
            nn.Linear(1152 + 16, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 10), nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, img, cmd):
        img_features = self.conv_layers(img)
        cmd = cmd.view(-1, 1)
        cmd_features = self.command_fc(cmd)
        combined = torch.cat((img_features, cmd_features), dim=1)
        return self.joint_fc(combined)


def train():

    print("\n" + "="*50)
    if DEVICE == "cuda":
        print(f" [V] ANTRENARE PE GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(" [X] ATENȚIE: GPU indisponibil, se folosește CPU (LENT!)")
    print("="*50 + "\n")
    
    print(f"--- INCEPERE ANTRENARE NAVIGATIE ---")
    
    dataset = CarlaNavDataset(DATASET_DIR)
    print(f" -> {len(dataset)} imagini cu comenzi gasite și pregătite.")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,        
        pin_memory=PIN_MEMORY,          
        persistent_workers=PERSISTENT_WORKERS, 
        prefetch_factor=PREFETCH_FACTOR, 
        drop_last=True                  
    )

    model = ConditionalNvidiaModel().to(DEVICE)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train() 
        total_loss = 0
        
        for batch, (imgs, cmds, labels) in enumerate(dataloader):
            imgs = imgs.to(DEVICE)
            cmds = cmds.to(DEVICE)
            labels = labels.to(DEVICE)

            pred = model(imgs, cmds)
            loss = loss_fn(pred.squeeze(), labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} — loss = {avg_loss:.6f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel salvat: '{MODEL_SAVE_PATH}'")

if __name__ == "__main__":
    train()