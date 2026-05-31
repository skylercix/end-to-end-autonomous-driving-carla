import os
import csv
import random
from datetime import datetime
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt 
import torch.nn.functional as F


DATASET_DIR = "dataset_traffic_processed"  
MODEL_SAVE_PATH = "model_nav_traffic.pth"  
GRAPH_SAVE_PATH = "training_history.png"
EXPERIMENT_LOG = "experiment_log.csv"

BATCH_SIZE = 64 
NUM_EPOCHS = 40          
LEARNING_RATE = 3e-4     
VAL_SPLIT = 0.15         

NUM_WORKERS = 0        
PREFETCH_FACTOR = None

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
        self.labels = [] 
        self.commands = []
        self.traffic_lights = []

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
                        throttle_val = float(row[2]) 
                        brake_val = float(row[3])    
                        cmd_val = int(row[4])       #GPS status
                        tl_val = int(row[5])        #TL status
                        
                        full_img_path = os.path.join(episode_path, img_name)
                        
                        if os.path.exists(full_img_path):
                            self.images.append(full_img_path)
                            self.labels.append([steer_val, throttle_val, brake_val]) 
                            self.commands.append(cmd_val)
                            self.traffic_lights.append(tl_val)
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
            img = Image.open(self.images[idx]).convert("RGB") 
            img = self.transform_pipeline(img)
            cmd = self.commands[idx]
            tl = self.traffic_lights[idx]
            targets = self.labels[idx]
            return img, torch.tensor(cmd, dtype=torch.float32), torch.tensor(tl, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)
        except Exception as e:
            return torch.zeros((3, 66, 200)), torch.tensor(0, dtype=torch.float32), torch.tensor(0, dtype=torch.float32), torch.zeros(3, dtype=torch.float32)


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

def train():
    print("\n" + "="*50)
    print(f" [V] ANTRENARE PE: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}")
    print("="*50 + "\n")
    
    #experiment log initialization
    if not os.path.exists(EXPERIMENT_LOG):
        header = ["timestamp", "model", "dataset_size", "epochs", "best_epoch",
                  "learning_rate", "best_val_loss", "final_train_loss", "notes"]
        historical_data = [
            ["2025-01-01 00:00", "CNN", "9485", "40", "~35",
             "3e-4", "0.03700", "-", "Prima antrenare CNN, dataset initial"],
            ["2025-01-15 00:00", "ViT (GPS concat)", "9485", "60", "44",
             "1e-4", "0.05080", "-", "ViT v1 - GPS concatenat la final dupa transformer"],
            ["2025-02-01 00:00", "ViT (GPS-as-Token)", "13000", "60", "60",
             "1e-4", "0.05100", "-", "ViT v2 - GPS ca token in secventa transformer"],
        ]
        with open(EXPERIMENT_LOG, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(historical_data)
        print(f"[LOG] Fisier {EXPERIMENT_LOG} creat cu {len(historical_data)} antrenari istorice.")

    dataset = CarlaNavDataset(DATASET_DIR)
    total_data = len(dataset)
    print(f" -> {total_data} imagini găsite.")
    
    if total_data == 0:
        return

    val_size = int(total_data * VAL_SPLIT)
    train_size = total_data - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f" -> Antrenare pe: {train_size} imagini | Validare pe: {val_size} imagini\n")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
                              
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = ConditionalNvidiaModel().to(DEVICE)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    
    
    history_train_loss = []
    history_val_loss = []

    for epoch in range(NUM_EPOCHS):
        model.train() 
        train_loss = 0
        
        for imgs, cmds, tls, labels in train_loader:
            imgs, cmds, tls, labels = imgs.to(DEVICE), cmds.to(DEVICE), tls.to(DEVICE), labels.to(DEVICE)

            pred = model(imgs, cmds, tls)
            loss = loss_fn(pred, labels) 

            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval() 
        val_loss = 0
        with torch.no_grad():
            for imgs, cmds, tls, labels in val_loader:
                imgs, cmds, tls, labels = imgs.to(DEVICE), cmds.to(DEVICE), tls.to(DEVICE), labels.to(DEVICE)
                pred = model(imgs, cmds, tls)
                loss = loss_fn(pred, labels)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        #plot dattas
        history_train_loss.append(avg_train_loss)
        history_val_loss.append(avg_val_loss)

        saved_flag = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            saved_flag = " [MODEL SALVAT]"

        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}{saved_flag}")

    print(f"\nAntrenare completa! Cu o eroare de validare de {best_val_loss:.5f}")

    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, NUM_EPOCHS + 1), history_train_loss, label='Train Loss', color='blue', linewidth=2)
    plt.plot(range(1, NUM_EPOCHS + 1), history_val_loss, label='Validation Loss', color='orange', linewidth=2, linestyle='--')
    
   
    best_epoch = history_val_loss.index(min(history_val_loss)) + 1
    plt.scatter(best_epoch, min(history_val_loss), color='red', s=100, zorder=5, label=f'Best Model (Epoca {best_epoch})')

   
    plt.xlabel("Epoca", fontsize=12)
    plt.ylabel("Eroare (MSE Loss)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=11)
    plt.show()

    #save in csv
    best_epoch = history_val_loss.index(min(history_val_loss)) + 1
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        "CNN",
        str(total_data),
        str(NUM_EPOCHS),
        str(best_epoch),
        str(LEARNING_RATE),
        f"{best_val_loss:.5f}",
        f"{history_train_loss[-1]:.5f}",
        f"NVIDIA CNN conditionat, MSE loss, Adam, ReduceLROnPlateau"
    ]
    with open(EXPERIMENT_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    print(f"[LOG] Experiment CNN salvat in {EXPERIMENT_LOG}")

if __name__ == "__main__":
    train()