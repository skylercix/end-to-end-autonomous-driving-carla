import os
import csv
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

DEVICE = "cpu"
DATASET_DIR = "dataset_small"



class CarlaDataset(Dataset):
    def __init__(self, root):
        self.images = []
        self.steering = []

        # Parcurgem toate ep
        for episode_folder in os.listdir(root):
            episode_path = os.path.join(root, episode_folder)
            if not os.path.isdir(episode_path):
                continue
            csv_path = os.path.join(episode_path, "controls.csv")
            if not os.path.exists(csv_path):
                print(f"Avertisment: Nu exista controls.csv in {episode_path}, se sare peste")
                continue
            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                for row in reader:
                    img_name = row[0]
                    steer = float(row[1])
                    self.images.append(os.path.join(episode_path, img_name))
                    self.steering.append(steer)

        if len(self.images) == 0:
            raise FileNotFoundError(f"Dataset gol in {root}!")

        self.transform = transforms.Compose([
            transforms.Resize((66, 200)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        img = self.transform(img)
        steer = torch.tensor(self.steering[idx], dtype=torch.float32)
        return img, steer


#model

class SmallNvidiaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 1 * 18, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        return self.net(x)


#taiin

def train():
    print("Using device:", DEVICE)

    dataset = CarlaDataset(DATASET_DIR)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SmallNvidiaModel().to(DEVICE)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("Dataset loaded cu", len(dataset), "imagini")

    for epoch in range(5):
        total = 0
        for batch, (imgs, labels) in enumerate(dataloader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            pred = model(imgs)
            loss = loss_fn(pred.squeeze(), labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"Epoch {epoch+1}/5 — loss = {total/len(dataloader):.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("Model salvat ca model.pth")


if __name__ == "__main__":
    train()
