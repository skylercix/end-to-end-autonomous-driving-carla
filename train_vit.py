import os
import csv
import random
import math
from datetime import datetime
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F


# === CONFIGURARE ===
DATASET_DIR = "dataset_traffic_processed"
MODEL_SAVE_PATH = "model_nav_traffic_vit.pth"
GRAPH_SAVE_PATH = "training_history_vit.png"
EXPERIMENT_LOG = "experiment_log.csv"

BATCH_SIZE = 64
NUM_EPOCHS = 60
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_EPOCHS = 5
VAL_SPLIT = 0.15

# Loss ponderat: steering e de 3x mai important
LOSS_WEIGHT_STEER = 2.0
LOSS_WEIGHT_THROTTLE = 1.0
LOSS_WEIGHT_BRAKE = 1.0

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


# === DATASET CU AUGMENTARE VIZUALA + GEOMETRICA ===
def convert_yuv(img):
    return img.convert("YCbCr")

class CarlaNavDataset(Dataset):
    def __init__(self, root, augment=False):
        self.images = []
        self.labels = []
        self.commands = []
        self.augment = augment

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
                        cmd_val = int(row[4])

                        full_img_path = os.path.join(episode_path, img_name)

                        if os.path.exists(full_img_path):
                            self.images.append(full_img_path)
                            self.labels.append([steer_val, throttle_val, brake_val])
                            self.commands.append(cmd_val)
                    except (ValueError, IndexError):
                        continue

        # Pipeline de baza
        self.base_pipeline = transforms.Compose([
            transforms.Lambda(convert_yuv),
            transforms.ToTensor(),
        ])

        # Pipeline cu augmentare vizuala + geometrica
        self.augment_pipeline = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.05
            ),
            transforms.RandomAffine(
                degrees=3,          # rotatie mica +-3 grade
                translate=(0.05, 0.05),  # translatie +-5%
                fill=0              # pixelii noi = negru
            ),
            transforms.RandomErasing(
                p=0.2,              # 20% sansa de ocluzie
                scale=(0.02, 0.08), # acopera 2-8% din imagine
                ratio=(0.3, 3.3),
            ),
            transforms.Lambda(convert_yuv),
            transforms.ToTensor(),
        ])

        # Pipeline doar geometrica (fara color, pentru RandomErasing care necesita tensor)
        # Restructuram: ColorJitter + Affine pe PIL, apoi YUV+ToTensor, apoi RandomErasing pe tensor
        self.augment_color_geo = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.05
            ),
            transforms.RandomAffine(
                degrees=3,
                translate=(0.05, 0.05),
                fill=0
            ),
        ])
        self.augment_post = transforms.Compose([
            transforms.Lambda(convert_yuv),
            transforms.ToTensor(),
            transforms.RandomErasing(
                p=0.2,
                scale=(0.02, 0.08),
                ratio=(0.3, 3.3),
            ),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.images[idx]).convert("RGB")

            # Augmentare cu 50% probabilitate (doar la antrenare)
            if self.augment and random.random() < 0.5:
                img = self.augment_color_geo(img)
                img = self.augment_post(img)
            else:
                img = self.base_pipeline(img)

            cmd = self.commands[idx]
            targets = self.labels[idx]
            return img, torch.tensor(cmd, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)
        except Exception:
            return torch.zeros((3, 66, 200)), torch.tensor(0, dtype=torch.float32), torch.zeros(3, dtype=torch.float32)


# === CONV STEM HYBRID VISION TRANSFORMER ===

class ConvStem(nn.Module):
    """
    Mini-CNN care inlocuieste patch embedding-ul brut.
    3 straturi conv cu BatchNorm -> extrag features locale (margini, texturi, linii)
    INAINTE ca transformerul sa le proceseze.
    
    Flux: 3x66x200 -> 48x33x100 -> 96x17x50 -> 128x9x25
    Rezultat: 9x25 = 225 tokens de 128 dimensiuni
    """
    def __init__(self, embed_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 48, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(48)
        
        self.conv2 = nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(96)
        
        self.conv3 = nn.Conv2d(96, embed_dim, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(embed_dim)
        
        self.act = nn.ReLU()
    
    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))   # -> (B, 48, 33, 100)
        x = self.act(self.bn2(self.conv2(x)))    # -> (B, 96, 17, 50)
        x = self.act(self.bn3(self.conv3(x)))    # -> (B, 128, 9, 25)
        return x


class TransformerBlock(nn.Module):
    """Bloc Transformer cu Pre-LayerNorm si stocare atentie."""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(dropout)
        )
        self.attn_weights = None

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, self.attn_weights = self.attn(x_norm, x_norm, x_norm, need_weights=True)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ConditionalViTModel(nn.Module):
    """
    Hybrid Vision Transformer cu Conv Stem + GPS ca Token.
    
    Conv Stem: 3 straturi CNN extrag features locale (margini, texturi)
    Rezultat: 9x25 = 225 patch-uri, fiecare deja cu informatii locale
    
    Secventa: [CLS] [GPS] [patch_1] ... [patch_225] = 227 tokens
    GPS participa la self-attention in TOATE straturile.
    """
    def __init__(self, img_h=66, img_w=200,
                 embed_dim=128, num_heads=4, num_layers=4, mlp_ratio=4, dropout=0.1):
        super().__init__()

        self.embed_dim = embed_dim

        #Conv Stem inlocuieste patch embedding-ul direct
        self.conv_stem = ConvStem(embed_dim)
        
        # Calculam numarul de patch-uri dupa conv stem
        # 66x200 -> stride 2 de 3 ori -> 9x25 = 225 patches
        self.num_patches_h = math.ceil(img_h / 8)   # 66/8 = 9 (cu padding)
        self.num_patches_w = math.ceil(img_w / 8)    # 200/8 = 25
        self.num_patches = self.num_patches_h * self.num_patches_w  # 225

        #CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        #GPS token embedding
        self.gps_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        #Positional embeddings: 1 CLS + 1 GPS + 225 patches = 227
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 2, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(dropout)

        #Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        #MLP head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 3)  # steer, throttle, brake
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, img, cmd):
        B = img.shape[0]

        #Conv Stem: extrage features locale
        # (B, 3, 66, 200) -> (B, 128, 9, 25)
        x = self.conv_stem(img)
        
        #Flatten spatial -> secventa de tokens
        # (B, 128, 9, 25) -> (B, 225, 128)
        x = x.flatten(2).transpose(1, 2)

        #CLS token
        cls = self.cls_token.expand(B, -1, -1)

        #GPS token
        cmd_onehot = F.one_hot(cmd.long(), num_classes=4).float()
        gps_token = self.gps_embed(cmd_onehot).unsqueeze(1)

        #secventa: [CLS] [GPS] [patch_1] ... [patch_225]
        x = torch.cat([cls, gps_token, x], dim=1)

        #positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        #transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # CLS output
        cls_out = x[:, 0]
        return self.head(cls_out)

    def get_attention_maps(self, layer_idx=-1):
        """Harta de atentie CLS -> patches."""
        block = self.blocks[layer_idx]
        if block.attn_weights is None:
            return None
        cls_attn = block.attn_weights[0, 0, 2:]  # skip CLS(0) si GPS(1)
        attn_map = cls_attn.reshape(self.num_patches_h, self.num_patches_w)
        attn_map = attn_map - attn_map.min()
        if attn_map.max() > 0:
            attn_map = attn_map / attn_map.max()
        return attn_map.detach().cpu().numpy()

    def get_gps_attention(self, layer_idx=-1):
        """Harta de atentie GPS -> patches."""
        block = self.blocks[layer_idx]
        if block.attn_weights is None:
            return None
        gps_attn = block.attn_weights[0, 1, 2:]  # GPS(1) -> patches(2:)
        attn_map = gps_attn.reshape(self.num_patches_h, self.num_patches_w)
        attn_map = attn_map - attn_map.min()
        if attn_map.max() > 0:
            attn_map = attn_map / attn_map.max()
        return attn_map.detach().cpu().numpy()


# === COSINE SCHEDULER CU WARMUP ===
class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = optimizer.param_groups[0]['lr']
        self.min_lr = min_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


# === WEIGHTED MSE LOSS ===
class WeightedMSELoss(nn.Module):
    """
    MSE loss cu ponderi diferite per output.
    Steering primeste greutate mai mare fiindca e critic pentru a sta pe banda.
    """
    def __init__(self, weight_steer=3.0, weight_throttle=1.0, weight_brake=1.0):
        super().__init__()
        self.weights = torch.tensor([weight_steer, weight_throttle, weight_brake])
    
    def forward(self, pred, target):
        weights = self.weights.to(pred.device)
        mse_per_output = (pred - target) ** 2  # (B, 3)
        weighted_mse = mse_per_output * weights  # (B, 3) broadcast
        return weighted_mse.mean()


#fisier pt statistici finale
def init_experiment_log(log_path):
    """
    Creeaza fisierul CSV daca nu exista si adauga datele.
    """
    if os.path.exists(log_path):
        return
    
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
    
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(historical_data)
    
    print(f"[LOG] Fisier {log_path} creat cu {len(historical_data)} antrenari istorice.")


def log_experiment(log_path, model_name, dataset_size, epochs, best_epoch,
                   lr, best_val_loss, final_train_loss, notes=""):
    """
    Adauga o linie noua in CSV-ul de experiment.
    """
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        model_name,
        str(dataset_size),
        str(epochs),
        str(best_epoch),
        str(lr),
        f"{best_val_loss:.5f}",
        f"{final_train_loss:.5f}",
        notes
    ]
    
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    
    print(f"[LOG] Experiment salvat in {log_path}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train():
    print("\n" + "=" * 60)
    print(f"  ANTRENARE Hybrid ViT (Conv Stem + GPS-as-Token)")
    print(f"  Device: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}")
    print("=" * 60 + "\n")

    
    init_experiment_log(EXPERIMENT_LOG)

    #augmentare doar pt antrenare
    dataset_aug = CarlaNavDataset(DATASET_DIR, augment=True)
    dataset_clean = CarlaNavDataset(DATASET_DIR, augment=False)
    total_data = len(dataset_clean)
    print(f" -> {total_data} imagini gasite.")

    if total_data == 0:
        return

    val_size = int(total_data * VAL_SPLIT)
    train_size = total_data - val_size

    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(range(total_data), [train_size, val_size], generator=generator)

    train_dataset = torch.utils.data.Subset(dataset_aug, train_indices.indices)
    val_dataset = torch.utils.data.Subset(dataset_clean, val_indices.indices)

    print(f" -> Antrenare pe: {train_size} imagini (cu augmentare) | Validare pe: {val_size} imagini")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = ConditionalViTModel().to(DEVICE)
    num_params = count_parameters(model)
    print(f"\n --- ARHITECTURA ---")
    print(f" -> Conv Stem: 3 -> 48 -> 96 -> 128 (3 straturi conv cu BN)")
    print(f" -> Tokens: 1 CLS + 1 GPS + {model.num_patches} patches = {model.num_patches + 2}")
    print(f" -> Embed dim: {model.embed_dim}, Layers: {len(model.blocks)}, Heads: {model.blocks[0].attn.num_heads}")
    print(f" -> Parametri totali: {num_params:,}")
    print(f"\n --- ANTRENARE ---")
    print(f" -> Loss ponderat: steer={LOSS_WEIGHT_STEER}x, throttle={LOSS_WEIGHT_THROTTLE}x, brake={LOSS_WEIGHT_BRAKE}x")
    print(f" -> LR: {LEARNING_RATE}, Warmup: {WARMUP_EPOCHS} epoci, Weight Decay: {WEIGHT_DECAY}")
    print(f" -> Augmentare: ColorJitter + RandomAffine(±3°, ±5%) + RandomErasing(20%)\n")

    loss_fn = WeightedMSELoss(LOSS_WEIGHT_STEER, LOSS_WEIGHT_THROTTLE, LOSS_WEIGHT_BRAKE)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineWarmupScheduler(opt, WARMUP_EPOCHS, NUM_EPOCHS)

    best_val_loss = float('inf')
    history_train_loss = []
    history_val_loss = []
    history_lr = []

    for epoch in range(NUM_EPOCHS):
        current_lr = scheduler.step(epoch)
        history_lr.append(current_lr)

        model.train()
        train_loss = 0

        for imgs, cmds, labels in train_loader:
            imgs, cmds, labels = imgs.to(DEVICE), cmds.to(DEVICE), labels.to(DEVICE)

            pred = model(imgs, cmds)
            loss = loss_fn(pred, labels)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, cmds, labels in val_loader:
                imgs, cmds, labels = imgs.to(DEVICE), cmds.to(DEVICE), labels.to(DEVICE)
                pred = model(imgs, cmds)
                loss = loss_fn(pred, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        history_train_loss.append(avg_train_loss)
        history_val_loss.append(avg_val_loss)

        saved_flag = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            saved_flag = " [MODEL SALVAT]"

        print(f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | LR: {current_lr:.6f} | "
              f"Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f}{saved_flag}")

    print(f"\nAntrenare completa! Eroare minima de validare: {best_val_loss:.5f} (epoca {best_epoch})")

    #salvare in csv
    log_experiment(
        EXPERIMENT_LOG,
        model_name="ViT Hybrid (Conv Stem + GPS Token)",
        dataset_size=total_data,
        epochs=NUM_EPOCHS,
        best_epoch=best_epoch,
        lr=LEARNING_RATE,
        best_val_loss=best_val_loss,
        final_train_loss=history_train_loss[-1],
        notes=f"Conv Stem 3->48->96->128, {model.num_patches} tokens, "
              f"weighted loss s={LOSS_WEIGHT_STEER} t={LOSS_WEIGHT_THROTTLE} b={LOSS_WEIGHT_BRAKE}, "
              f"aug: ColorJitter+Affine+Erasing"
    )

    #grafuri
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Antrenare Hybrid ViT (Conv Stem + GPS-as-Token)", fontsize=14, fontweight='bold')

    ax1.plot(range(1, NUM_EPOCHS + 1), history_train_loss, label='Train Loss', color='blue', linewidth=2)
    ax1.plot(range(1, NUM_EPOCHS + 1), history_val_loss, label='Val Loss', color='orange', linewidth=2, linestyle='--')
    best_epoch_idx = history_val_loss.index(min(history_val_loss)) + 1
    ax1.scatter(best_epoch_idx, min(history_val_loss), color='red', s=100, zorder=5,
                label=f'Best Model (Epoca {best_epoch_idx})')
    ax1.set_xlabel("Epoca", fontsize=12)
    ax1.set_ylabel("Eroare (Weighted MSE Loss)", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.7)

    ax2.plot(range(1, NUM_EPOCHS + 1), history_lr, color='green', linewidth=2)
    ax2.set_xlabel("Epoca", fontsize=12)
    ax2.set_ylabel("Learning Rate", fontsize=12)
    ax2.set_title("Cosine Schedule cu Warmup")
    ax2.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig(GRAPH_SAVE_PATH, dpi=150)
    print(f"Grafic salvat: {GRAPH_SAVE_PATH}")
    plt.show()


if __name__ == "__main__":
    train()