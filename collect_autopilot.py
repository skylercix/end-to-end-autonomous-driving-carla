import carla
import time
import os
import csv
import numpy as np
from PIL import Image
import queue
import pygame
from pygame.locals import K_SPACE, K_ESCAPE, K_m, K_w, K_a, K_s, K_d
import random
import sys
import glob
import math  


try:
    sys.path.append(glob.glob('../carla')[0])
except IndexError:
    pass

from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import RoadOption

ROOT_SAVE_FOLDER = "dataset_manual" 
os.makedirs(ROOT_SAVE_FOLDER, exist_ok=True)

def map_command(road_option):
    if road_option == RoadOption.LEFT: return 1
    elif road_option == RoadOption.RIGHT: return 2
    elif road_option == RoadOption.STRAIGHT: return 3
    else: return 0 

def cleanup_actors(world):
    actors = world.get_actors()
    for actor in actors.filter('vehicle.*'):
        actor.destroy()
    for actor in actors.filter('sensor.*'):
        actor.destroy()
    time.sleep(1) 

def get_next_episode_path(root_folder):
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    existing_folders = [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d)) and d.startswith("episode_")]
    max_idx = -1
    for folder in existing_folders:
        try:
            idx = int(folder.split("_")[1])
            if idx > max_idx: max_idx = idx
        except: continue
    return os.path.join(root_folder, f"episode_{max_idx + 1:03d}")

def save_image(image, control, command, episode_path, buffer_controls):
    image.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1] 
    pil_image = Image.fromarray(array)
    filename = f"{image.frame}.png"
    pil_image.save(os.path.join(episode_path, filename))
    buffer_controls.append([filename, control.steer, control.throttle, control.brake, command])

def save_csv(episode_path, buffer_controls):
    if not buffer_controls:
        try: os.rmdir(episode_path)
        except: pass
        return
    csv_path = os.path.join(episode_path, "controls_nav.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "steer", "throttle", "brake", "command"])
        writer.writerows(buffer_controls)
    print(f" Salvat: {os.path.basename(episode_path)} | {len(buffer_controls)} imagini.")

def main():
    pygame.init()
    
    display = pygame.display.set_mode((450, 400))
    pygame.display.set_caption("Colectare Hibridă | SPACE=Rec | M=Manual")
    font = pygame.font.SysFont("Arial", 18)

    print("Se conectează la simulator...")
    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0) 
    world = client.get_world()

    current_map_name = world.get_map().name
    if not current_map_name.endswith('Town01'):
        world = client.load_world('Town01')

    cleanup_actors(world)
    blueprint_library = world.get_blueprint_library()

    vehicle_bp = blueprint_library.filter("model3")[0]
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = np.random.choice(spawn_points) if spawn_points else carla.Transform()
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if not vehicle: return
    time.sleep(1.0)

    # --- SETUP AGENT ---
    agent = BasicAgent(vehicle, target_speed=20)
    agent.ignore_traffic_lights(active=True)
    agent.ignore_stop_signs(active=True)
    
    current_wp = world.get_map().get_waypoint(vehicle.get_location())
    next_wps = current_wp.next(100.0)
    if next_wps: agent.set_destination(next_wps[0].transform.location)
    else: agent.set_destination(random.choice(spawn_points).location)

    # --- SETUP CAMERA ---
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", "320")
    camera_bp.set_attribute("image_size_y", "240")
    camera_bp.set_attribute("fov", "90")
    camera_bp.set_attribute("sensor_tick", "0.1") 
    cam_transform = carla.Transform(carla.Location(x=1.5, z=1.4), carla.Rotation(pitch=-15.0))
    camera = world.spawn_actor(camera_bp, cam_transform, attach_to=vehicle)

    image_queue = queue.Queue()
    camera.listen(image_queue.put)

    spectator = world.get_spectator()
    clock = pygame.time.Clock()

    
    is_recording = False
    current_episode_path = None
    buffer_controls = [] 
    
    autopilot_enabled = True
    m_pressed_last_frame = False
    manual_steer = 0.0 

    try:
        while True:
            clock.tick(60)
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            
            if keys[K_ESCAPE]: break


            #AUTOPILOT/MANUAL
 
            if keys[K_m] and not m_pressed_last_frame:
                autopilot_enabled = not autopilot_enabled
                print(f"\n>>> MOD CONDUS: {'🤖 AUTOPILOT' if autopilot_enabled else '🕹️ MANUAL (WASD)'} <<<")
            m_pressed_last_frame = keys[K_m]

            
            #GPS
            
            if agent.done():
                current_loc = vehicle.get_location()
                new_dest = random.choice(spawn_points)
                while new_dest.location.distance(current_loc) < 80.0:
                    new_dest = random.choice(spawn_points)
                agent.set_destination(new_dest.location)
            
            auto_control = agent.run_step()
            current_road_option = agent.get_local_planner().target_road_option
            current_command = map_command(current_road_option)

            
            if autopilot_enabled:
                control_to_apply = auto_control
                manual_steer = auto_control.steer 
            else:
                control_to_apply = carla.VehicleControl()
                
               
                if keys[K_w]: control_to_apply.throttle = 0.5
                elif keys[K_s]: control_to_apply.brake = 1.0
                else: control_to_apply.throttle = 0.0
                
                
                steer_speed = 0.05
                if keys[K_a]:
                    manual_steer = max(-1.0, manual_steer - steer_speed)
                elif keys[K_d]:
                    manual_steer = min(1.0, manual_steer + steer_speed)
                else:
                    if manual_steer > 0: manual_steer = max(0.0, manual_steer - steer_speed)
                    elif manual_steer < 0: manual_steer = min(0.0, manual_steer + steer_speed)
                
                control_to_apply.steer = manual_steer

            vehicle.apply_control(control_to_apply)

          
            if keys[K_SPACE] and not is_recording:
                is_recording = True
                current_episode_path = get_next_episode_path(ROOT_SAVE_FOLDER)
                os.makedirs(current_episode_path, exist_ok=True)
                buffer_controls = []
                time.sleep(0.2) 
            elif not keys[K_SPACE] and is_recording:
                is_recording = False
                save_csv(current_episode_path, buffer_controls)

            
            bg_color = (200, 0, 0) if is_recording else (0, 0, 0)
            display.fill(bg_color)
            
            driver_str = "AUTO" if autopilot_enabled else "MANUAL"
            cmd_str = ["LANE", "LEFT", "RIGHT", "STRAIGHT"][current_command]
            
            text_1 = font.render(f"Driver: {driver_str} | REC: {'DA' if is_recording else 'NU'}", True, (255,255,255))
            text_2 = font.render(f"GPS CMD: {cmd_str} | Steer: {control_to_apply.steer:.2f}", True, (255,255,255))
            text_3 = font.render(f"[M] Toggle Manual | [SPACE] Hold to Record", True, (150,150,150))
            text_4 = font.render(f"RADAR GPS (2D) \/", True, (255, 255, 0))
            
            display.blit(text_1, (10, 10))
            display.blit(text_2, (10, 40))
            display.blit(text_3, (10, 70))
            display.blit(text_4, (155, 120))

            if vehicle.is_alive:
                v_transform = vehicle.get_transform()
                v_x = v_transform.location.x
                v_y = v_transform.location.y
                
                v_yaw = math.radians(v_transform.rotation.yaw)
                
                route_trace = list(agent.get_local_planner()._waypoints_queue)
                
                
                radar_center_x, radar_center_y = 225, 350
                
               
                pygame.draw.circle(display, (0, 150, 255), (radar_center_x, radar_center_y), 6)

                for wp, _ in route_trace[:30]:
                    w_x = wp.transform.location.x
                    w_y = wp.transform.location.y
                    
            
                    dx = w_x - v_x
                    dy = w_y - v_y
                    
                   
                    rel_x = dx * math.cos(v_yaw) + dy * math.sin(v_yaw)
                    rel_y = -dx * math.sin(v_yaw) + dy * math.cos(v_yaw)
                    
                    
                    scale = 4  
                    screen_x = int(radar_center_x + rel_y * scale)
                    screen_y = int(radar_center_y - rel_x * scale) 
                    
                    
                    if 0 <= screen_x <= 450 and 0 <= screen_y <= 400:
                        pygame.draw.circle(display, (0, 255, 0), (screen_x, screen_y), 3)

            pygame.display.flip()
            

            if vehicle.is_alive:
                t = vehicle.get_transform()
                spectator.set_transform(carla.Transform(
                    t.location - 6 * t.get_forward_vector() + carla.Location(z=3.0),
                    carla.Rotation(pitch=-15.0, yaw=t.rotation.yaw)
                ))

            try:
                last_image = None
                while not image_queue.empty(): last_image = image_queue.get_nowait()
                
                if is_recording and last_image is not None:
                    speed = vehicle.get_velocity()
                    speed_kmh = (3.6 * np.sqrt(speed.x**2 + speed.y**2 + speed.z**2))
                    if speed_kmh > 1.0:
                        save_image(last_image, control_to_apply, current_command, current_episode_path, buffer_controls)
            except queue.Empty: pass

    finally:
        if is_recording and buffer_controls:
            save_csv(current_episode_path, buffer_controls)
        if camera: camera.destroy()
        if vehicle: vehicle.destroy()
        pygame.quit()
        print("Script oprit.")

if __name__ == "__main__":
    main()