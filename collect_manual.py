import carla
import time
import os
import csv
import numpy as np
from PIL import Image
import queue
import pygame
from pygame.locals import K_w, K_a, K_s, K_d, K_SPACE, K_ESCAPE


ROOT_SAVE_FOLDER = "dataset_manual" 
os.makedirs(ROOT_SAVE_FOLDER, exist_ok=True)


def get_next_episode_path(root_folder):
    """
    Scaneaza folderul si returneaza calea pentru urmatorul episod disponibil
    """
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    
    
    existing_folders = [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d)) and d.startswith("episode_")]
    
    max_idx = -1
    for folder in existing_folders:
        try:
          
            idx = int(folder.split("_")[1])
            if idx > max_idx:
                max_idx = idx
        except (IndexError, ValueError):
            continue
    

    next_idx = max_idx + 1
    
    new_folder_name = f"episode_{next_idx:03d}"
    return os.path.join(root_folder, new_folder_name)

def save_image(image, control, episode_path, buffer_controls):
    """
    Salveaza imaginea si adauga datele in buffer.
    """
    image.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]
    pil_image = Image.fromarray(array)
    
    filename = f"{image.frame}.png"
    
    pil_image.save(os.path.join(episode_path, filename))
    
    buffer_controls.append([filename, control.steer, control.throttle, control.brake])

def save_csv(episode_path, buffer_controls):
    """
    Scrie CSV-ul pe disc pentru sesiunea curenta.
    """
    if not buffer_controls:
        
        try:
            os.rmdir(episode_path)
            print(f"Sters folder gol: {episode_path}")
        except: pass
        return

    csv_path = os.path.join(episode_path, "controls.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "steer", "throttle", "brake"])
        writer.writerows(buffer_controls)
    print(f" Salvat: {os.path.basename(episode_path)} | {len(buffer_controls)} imagini.")

def main():
    pygame.init()
    display = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("CLICK AICI -> WASD Drive | SPACE Record")
    font = pygame.font.SysFont("Arial", 20)

    print("Se conectează la simulator...")
    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)
    #client.load_world('Town04') #tw4 si tw1 merg
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # ---SPAWN VEHICUL---
    vehicle_bp = blueprint_library.filter("model3")[0]
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = np.random.choice(spawn_points) if spawn_points else carla.Transform()
    
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle is None:
        print("Eroare la spawn vehicle.")
        return

    # ---SPAWN CAMERA---
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", "320")
    camera_bp.set_attribute("image_size_y", "240")
    camera_bp.set_attribute("fov", "90")
    camera_bp.set_attribute("sensor_tick", "0.1") # 10 FPS
    
    cam_transform = carla.Transform(carla.Location(x=1.5, z=1.4), carla.Rotation(pitch=-15.0))
    camera = world.spawn_actor(camera_bp, cam_transform, attach_to=vehicle)

    image_queue = queue.Queue()
    camera.listen(image_queue.put)

    spectator = world.get_spectator()
    clock = pygame.time.Clock()

    # ---VARIABILE DE STARE---
    is_recording = False
    current_episode_path = None
    buffer_controls = [] 

    print("\n--- BATCH RECORDING (Episodes) ---")
    print("Hold space to record an episode.")
    print("----------------------------------\n")

    try:
        while True:
            clock.tick(60)
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            
            if keys[K_ESCAPE]:
                break

            
            control = carla.VehicleControl()
            if keys[K_w]: control.throttle = 0.6
            if keys[K_s]: control.brake = 1.0
            if keys[K_a]: control.steer = -0.6
            elif keys[K_d]: control.steer = 0.6
            vehicle.apply_control(control)

            
            if keys[K_SPACE] and not is_recording:
                is_recording = True
                
               
                current_episode_path = get_next_episode_path(ROOT_SAVE_FOLDER)
                os.makedirs(current_episode_path, exist_ok=True)
                
                buffer_controls = []
                folder_name = os.path.basename(current_episode_path)
                
                pygame.display.set_caption(f" REC: {folder_name}")
                display.fill((200, 0, 0)) 

            
            elif keys[K_SPACE] and is_recording:
                pass 

            
            elif not keys[K_SPACE] and is_recording:
                is_recording = False
                
                #
                save_csv(current_episode_path, buffer_controls)
                
                pygame.display.set_caption("Stby - Hold SPACE for NEW episode")
                display.fill((0, 0, 0)) 

            elif not is_recording:
                 display.fill((0, 0, 0))

            
            folder_display = os.path.basename(current_episode_path) if current_episode_path else "Ready"
            status_text = f"Folder: {folder_display} | Steer: {control.steer:.2f}"
            text_surface = font.render(status_text, True, (255, 255, 255))
            display.blit(text_surface, (10, 10))
            pygame.display.flip()

            
            if vehicle.is_alive:
                t = vehicle.get_transform()
                cam_loc = t.location - 6 * t.get_forward_vector() + carla.Location(z=3.0)
                cam_rot = t.rotation
                cam_rot.pitch = -15.0
                spectator.set_transform(carla.Transform(cam_loc, cam_rot))

            
            try:
                last_image = None
                while not image_queue.empty():
                    last_image = image_queue.get_nowait()
                
                if is_recording and last_image is not None:
                    current_control = vehicle.get_control()
                    
                    
                    speed = vehicle.get_velocity()
                    speed_kmh = (3.6 * np.sqrt(speed.x**2 + speed.y**2 + speed.z**2))
                    
                    if speed_kmh > 1.0:
                        save_image(last_image, current_control, current_episode_path, buffer_controls)
                    
            except queue.Empty:
                pass

    finally:
        if is_recording and buffer_controls:
            save_csv(current_episode_path, buffer_controls)

        if camera: camera.destroy()
        if vehicle: vehicle.destroy()
        pygame.quit()
        print("Script oprit.")

if __name__ == "__main__":
    main()