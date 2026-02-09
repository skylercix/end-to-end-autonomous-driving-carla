import carla
import time
import os
import csv
import numpy as np
from PIL import Image
import keyboard
import queue


SAVE_FOLDER = "dataset_small"
NUM_EPISODES = 10
EPISODE_DURATION = 50


os.makedirs(SAVE_FOLDER, exist_ok=True)

# queue pt salvarea pozelor
data_queue = queue.Queue()

def sensor_callback(image, vehicle):
    """
    Callback: Doar pune datele in queuue.
    Nu salvam pe disc aici pentru a evita crash-ul.
    """
    if not vehicle.is_alive: return
    try:
        control = vehicle.get_control()
        
        data_queue.put((image, control))
    except:
        pass

def save_data_from_queue(episode_folder, buffer_controls):
    """
    Functie care scrie pe disc
    """
    while not data_queue.empty():
        try:
            image, control = data_queue.get()
            
            # Procesare imagine
            image.convert(carla.ColorConverter.Raw)
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))[:, :, :3]
            pil_image = Image.fromarray(array)
            
            filename = f"{image.frame}.png"
            pil_image.save(os.path.join(episode_folder, filename))
            
            # Adaugam in buffer pentru CSV
            buffer_controls.append([filename, control.steer, control.throttle, control.brake])
            
            data_queue.task_done()
        except Exception as e:
            print(f"Eroare la scriere fisier: {e}")

def run_episode(client, episode_num):
    # Curatam coada veche
    with data_queue.mutex:
        data_queue.queue.clear()

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter("model3")[0]

    # Retry spawn
    spawn_points = world.get_map().get_spawn_points()
    vehicle = None
    attempts = 0
    while vehicle is None and attempts < 20:
        spawn_point = np.random.choice(spawn_points)
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        attempts += 1
        if vehicle is None:
            time.sleep(0.5)

    if vehicle is None:
        print(f"Episode {episode_num}: Vehicle spawn failed after 20 attempts!")
        return False

    print(f"Episode {episode_num}: Vehicle spawned!")

    # pentru camera
    spectator = world.get_spectator()
    follow_camera = False 

    # spawnam camera la locul in care isi da spawn si masina
    transform = vehicle.get_transform()
    # Pozitionarea camerei (nu prea merge)
    loc_init = transform.location - transform.get_forward_vector() * 5.0 + carla.Location(z=3.0)
    rot_init = transform.rotation
    rot_init.pitch = -15.0
    spectator.set_transform(carla.Transform(loc_init, rot_init))

    def update_spectator():
        # Se apeleaza doar daca follow_camera este True
        if vehicle.is_alive:
            t = vehicle.get_transform()
            loc = t.location - t.get_forward_vector() * 6.0 + carla.Location(z=3.0)
            rot = t.rotation
            rot.pitch = -15.0
            spectator.set_transform(carla.Transform(loc, rot))
    # -----------------------------------

    # Camera attach la vehicul
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", "320")
    camera_bp.set_attribute("image_size_y", "240")
    camera_bp.set_attribute("fov", "90")
   # 10 fps
    camera_bp.set_attribute("sensor_tick", "0.1") 
    
    cam_transform = carla.Transform(carla.Location(x=1.5, z=2.0))
    camera = world.spawn_actor(camera_bp, cam_transform, attach_to=vehicle)

    buffer_controls = []
    episode_folder = os.path.join(SAVE_FOLDER, f"episode_{episode_num:03d}")
    os.makedirs(episode_folder, exist_ok=True)
    
    # fast callback-ul
    camera.listen(lambda image: sensor_callback(image, vehicle))

    vehicle.set_autopilot(True)

    print(f"Episode {episode_num}: Collecting data for {EPISODE_DURATION} seconds...")

    start_time = time.time()
    try:
        while time.time() - start_time < EPISODE_DURATION:
            if not vehicle.is_alive:
                break

            # V pentru third person view la camera
            if keyboard.is_pressed('v'):
                follow_camera = not follow_camera
                print(f" Camera Follow: {follow_camera}")
                time.sleep(0.3) # Debounce
            
           
            if follow_camera:
                update_spectator()
            
            
            save_data_from_queue(episode_folder, buffer_controls)
            
            
            world.wait_for_tick()

    finally:
        # Cleanup ca sa nu dea crash
        try:
            if camera is not None:
                camera.stop()
                camera.destroy()
        except: pass
        try:
            if vehicle is not None:
                vehicle.destroy()
        except: pass

        # Salvam ce e in queue
        save_data_from_queue(episode_folder, buffer_controls)

    # Salvam CSV
    csv_path = os.path.join(episode_folder, "controls.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "steer", "throttle", "brake"])
        writer.writerows(buffer_controls)

    print(f"Episode {episode_num}: Data collection done.")
    return True


def main():
    client = carla.Client("localhost", 2000)
   
    client.set_timeout(30.0)

    ep = 0
    while ep < NUM_EPISODES:
        try:
            success = run_episode(client, ep)
            if success:
                ep += 1
            else:
                print(f"Episode {ep} failed, retrying...")
            time.sleep(2) 
        except Exception as e:
            print(f"Episode {ep} failed with exception:", e)
            time.sleep(2)

if __name__ == "__main__":
    main()