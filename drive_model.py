import carla
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import time
import keyboard
import random

MODEL_PATH = "model.pth"
DEVICE = "cpu"


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


transform = transforms.Compose([
    transforms.Resize((66, 200)),
    transforms.ToTensor(),
])

def image_to_tensor(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]
    pil = Image.fromarray(array)
    t = transform(pil).unsqueeze(0).to(DEVICE)
    return t


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    
    model = SmallNvidiaModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model încărcat!")

    
    camera = None
    latest_image = None
    
    def grab_image(image):
        nonlocal latest_image
        latest_image = image

    vehicle = None
    respawn_requested = True  # spawn initial
    spawn_points = world.get_map().get_spawn_points()
    spectator = world.get_spectator()

    try:
        while True:
            # r respawn
            if keyboard.is_pressed('r'):
                respawn_requested = True

           
            if respawn_requested or vehicle is None:
                if vehicle is not None:
                    vehicle.destroy()
                    print("Vehicle destroyed. Respawning...")
                
                
                if camera is not None:
                    camera.destroy()
                    camera = None
                    latest_image = None 

                # random spawn point
                spawn_point = random.choice(spawn_points)
                vehicle_bp = blueprint_library.filter("model3")[0]
                vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
                
                if vehicle is None:
                    print("Spawn failed, retrying...")
                    continue 

                
                print(f"Vehicle spawned at {spawn_point.location}.")
                
                camera_bp = blueprint_library.find("sensor.camera.rgb")
                camera_bp.set_attribute("image_size_x", "320")
                camera_bp.set_attribute("image_size_y", "240")
                cam_transform = carla.Transform(carla.Location(x=1.5, z=2))
            
                camera = world.spawn_actor(camera_bp, cam_transform, attach_to=vehicle)
                camera.listen(grab_image)
                

                
                spectator.set_transform(carla.Transform(
                    spawn_point.location + carla.Location(z=6),  
                    spawn_point.rotation
                ))

                respawn_requested = False
                
            
            if latest_image is None:
                continue

            img_tensor = image_to_tensor(latest_image)
            with torch.no_grad():
                steer = float(model(img_tensor)[0])

            control = carla.VehicleControl()
            control.throttle = 0.35  # viteza constantă
            control.steer = steer
            vehicle.apply_control(control)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Oprit de utilizator.")

    finally:
        
        if camera is not None:
            camera.stop()
            camera.destroy()
        if vehicle is not None:
            vehicle.destroy()

if __name__ == "__main__":
    main()