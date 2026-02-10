import os
import csv
import random
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF



DATASET_DIR = "dataset_manual"
MODEL_SAVE_PATH = "model.pth"

BATCH_SIZE = 64         
NUM_EPOCHS = 50         
LEARNING_RATE = 1e-4    

NUM_WORKERS = 4         
PREFETCH_FACTOR = 2     


if torch.cuda.is_available():
    DEVICE = "cuda"
    PIN_MEMORY = True   
    PERSISTENT_WORKERS = True if NUM_WORKERS > 0 else False
else:
    DEVICE = "cpu"
    PIN_MEMORY = False
    PERSISTENT_WORKERS = False

# ==========================================
# FUNCȚII TRANSFORMARE
# ==========================================
def crop_img(img):
    return img.crop((0, 80, 320, 240))

def convert_yuv(img):
    return img.convert("YCbCr")

# ==========================================
#  DATASET
# ==========================================
class CarlaDataset(Dataset):
    def __init__(self, root):
        self.images = []
        self.steering = []

        if not os.path.exists(root):
             raise FileNotFoundError(f"Folderul {root} nu exista!")

       
        print(f"Se scaneaza dataset-ul in: {root} ...")
        
        for episode_folder in os.listdir(root):
            episode_path = os.path.join(root, episode_folder)
            if not os.path.isdir(episode_path):
                continue
            
            csv_path = os.path.join(episode_path, "controls.csv")
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
                        full_img_path = os.path.join(episode_path, img_name)
                        
                        if os.path.exists(full_img_path):
                            self.images.append(full_img_path)
                            self.steering.append(steer_val)
                    except ValueError:
                        continue 

        print(f" {len(self.images)} imagini valide gasite.")

        if len(self.images) == 0:
            raise RuntimeError(f"Nu exista nicio imagine valida in {root}!")

        self.transform_pipeline = transforms.Compose([
            transforms.Lambda(crop_img),        
            transforms.Lambda(convert_yuv),     
            transforms.Resize((66, 200)),       
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.images[idx]).convert("RGB")
        except Exception as e:
            print(f"Eroare: {e}")
            return torch.zeros((3, 66, 200)), torch.tensor(0.0, dtype=torch.float32)

        steer = self.steering[idx]

        if random.random() > 0.5:
            img = TF.hflip(img)
            steer = -steer

        img = self.transform_pipeline(img)
        return img, torch.tensor(steer, dtype=torch.float32)

# ==========================================
#  MODEL
# ==========================================
class SmallNvidiaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.ReLU(),
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

# ==========================================
#  TRAIN LOOP
# ==========================================
def train():
  
    print(f"\n--- INCEPERE ANTRENARE ---")
    if DEVICE == "cuda":
        print(f" GPU Activat: {torch.cuda.get_device_name(0)}")
    else:
        print(" Se foloseste CPU.")
        
    print(f"Batch Size: {BATCH_SIZE} | Workers: {NUM_WORKERS}")

    dataset = CarlaDataset(DATASET_DIR)
    
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

    model = SmallNvidiaModel().to(DEVICE)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n Se începe antrenarea...")

    for epoch in range(NUM_EPOCHS):
        model.train() 
        total_loss = 0
        
        for batch, (imgs, labels) in enumerate(dataloader):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            pred = model(imgs)
            loss = loss_fn(pred.squeeze(), labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} — loss = {avg_loss:.5f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel salvat cu succes: '{MODEL_SAVE_PATH}'")

if __name__ == "__main__":
    train()