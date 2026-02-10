import carla
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import time
import keyboard
import random

# --- CONFIGURARE ---
MODEL_PATH = "model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Rulez pe dispozitiv: {DEVICE}")

# --- 1. MODELUL (Trebuie sa fie IDENTIC cu cel din train.py) ---
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

# --- 2. TRANSFORMARI (Exact ca in train.py) ---
def crop_img(img):
    return img.crop((0, 80, 320, 240))

def convert_yuv(img):
    return img.convert("YCbCr")

transform_pipeline = transforms.Compose([
    transforms.Lambda(crop_img),
    transforms.Lambda(convert_yuv),
    transforms.Resize((66, 200)),
    transforms.ToTensor(),
])

def image_to_tensor(image):
    """
    Converteste imaginea raw din CARLA (BGRA) in Tensor compatibil cu modelul (RGB -> YUV)
    """
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    
    # CARLA da BGRA, noi vrem RGB.
    # Luam primele 3 canale [:3] si le inversam ordinea [::-1] (BGR -> RGB)
    array = array[:, :, :3][:, :, ::-1] 
    
    pil = Image.fromarray(array)
    
    # Aplicam pipeline-ul (Crop -> YUV -> Resize -> Tensor)
    t = transform_pipeline(pil).unsqueeze(0).to(DEVICE)
    return t

# --- 3. MAIN LOOP ---
def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # Incarcam modelul
    model = SmallNvidiaModel().to(DEVICE)
    
    # Incarcare robusta a greutatilor (Weights)
    if DEVICE == "cpu":
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(MODEL_PATH))
        
    model.eval() # Modul de evaluare (fara dropout, etc)
    print(" Model încărcat cu succes!")
    
    camera = None
    vehicle = None
    latest_image = None
    
    def grab_image(image):
        nonlocal latest_image
        latest_image = image

    respawn_requested = True
    spawn_points = world.get_map().get_spawn_points()
    spectator = world.get_spectator()
    
    follow_mode = True 
    print("\n--- COMENZI ---")
    print("[V] - Schimba camera (Follow / Free)")
    print("[R] - Reseteaza masina")
    print("[Ctrl+C] - Iesire")

    try:
        while True:
            # --- INPUT ---
            if keyboard.is_pressed('v'):
                follow_mode = not follow_mode
                print(f"Camera Mode: {'FOLLOW' if follow_mode else 'FREE'}")
                time.sleep(0.3)

            if keyboard.is_pressed('r'):
                respawn_requested = True

            # --- RESPAWN LOGIC ---
            if respawn_requested or vehicle is None or not vehicle.is_alive:
                # Curatenie
                if vehicle: vehicle.destroy()
                if camera: camera.destroy()
                
                # Spawn nou
                spawn_point = random.choice(spawn_points)
                vehicle_bp = blueprint_library.filter("model3")[0]
                vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
                
                if vehicle:
                    print(f"Masina spawnata la: {spawn_point.location}")
                    
                    # Setup Camera
                    camera_bp = blueprint_library.find("sensor.camera.rgb")
                    camera_bp.set_attribute("image_size_x", "320")
                    camera_bp.set_attribute("image_size_y", "240")
                    # Camera pozitionata pe capota/parbriz
                    cam_transform = carla.Transform(carla.Location(x=1.5, z=2.0))
                    camera = world.spawn_actor(camera_bp, cam_transform, attach_to=vehicle)
                    camera.listen(grab_image)
                    
                    # Resetam fizica putin ca sa nu cada prin harta
                    vehicle.set_simulate_physics(True)
                    respawn_requested = False
                else:
                    print("Failed to spawn. Retrying...")
                    time.sleep(0.5)
                
                continue # Sarim o tura de bucla pana se initializeaza totul

            # --- CONDUS AUTONOM ---
            if latest_image is not None:
                # 1. Procesam imaginea
                img_tensor = image_to_tensor(latest_image)
                
                # 2. Modelul prezice
                with torch.no_grad():
                    steer = float(model(img_tensor)[0])

                # 3. Aplicam comanda
                control = carla.VehicleControl()
                control.throttle = 0.35 
                control.steer = steer
                vehicle.apply_control(control)

                # 4. Camera Follow (Spectator)
                if follow_mode:
                    t = vehicle.get_transform()
                    spectator.set_transform(carla.Transform(
                        t.location + carla.Location(z=10, x=-6),
                        carla.Rotation(pitch=-35, yaw=t.rotation.yaw)
                    ))

            time.sleep(0.05) # ~20 FPS loop

    except KeyboardInterrupt:
        print("\nOprit.")

    finally:
        print("Curatare...")
        if camera: camera.destroy()
        if vehicle: vehicle.destroy()
        print("Exit!")

if __name__ == "__main__":
    main()