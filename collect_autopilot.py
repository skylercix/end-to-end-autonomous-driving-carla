import carla
import time
import os
import csv
import numpy as np
from PIL import Image
import queue
import pygame
from pygame.locals import K_SPACE, K_ESCAPE, K_m, K_w, K_a, K_s, K_d, K_n, K_r
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

ROOT_SAVE_FOLDER = "dataset_traffic" 
os.makedirs(ROOT_SAVE_FOLDER, exist_ok=True)

# =====================================================================
#                        Weather presets
# =====================================================================

WEATHER_PRESETS = {
    "ZI_SENINA": carla.WeatherParameters(
        sun_altitude_angle=70.0,       
        cloudiness=10.0,               
        precipitation=0.0,             
        precipitation_deposits=0.0,    
        wind_intensity=10.0,
        fog_density=0.0,               
        fog_distance=0.0,
        wetness=0.0,                   
        sun_azimuth_angle=0.0
    ),
    "INNORAT": carla.WeatherParameters(
        sun_altitude_angle=50.0,
        cloudiness=80.0,               
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=30.0,
        fog_density=0.0,
        fog_distance=0.0,
        wetness=0.0,
        sun_azimuth_angle=90.0
    ),
    "PLOAIE_USOARA": carla.WeatherParameters(
        sun_altitude_angle=40.0,
        cloudiness=70.0,
        precipitation=30.0,            
        precipitation_deposits=30.0,   
        wind_intensity=40.0,
        fog_density=5.0,
        fog_distance=0.0,
        wetness=40.0,                  
        sun_azimuth_angle=180.0
    ),
    "PLOAIE_PUTERNICA": carla.WeatherParameters(
        sun_altitude_angle=30.0,
        cloudiness=90.0,
        precipitation=70.0,            
        precipitation_deposits=70.0,  
        wind_intensity=70.0,
        fog_density=10.0,
        fog_distance=0.0,
        wetness=80.0,
        sun_azimuth_angle=270.0
    ),
    "CEATA": carla.WeatherParameters(
        sun_altitude_angle=45.0,
        cloudiness=50.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=5.0,
        fog_density=40.0,              
        fog_distance=30.0,             
        wetness=20.0,
        sun_azimuth_angle=45.0
    ),
    "APUS": carla.WeatherParameters(
        sun_altitude_angle=10.0,       
        cloudiness=20.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=10.0,
        fog_density=5.0,
        fog_distance=0.0,
        wetness=0.0,
        sun_azimuth_angle=220.0        
    ),
}

WEATHER_NAMES = list(WEATHER_PRESETS.keys())


def map_command(road_option):
    if road_option == RoadOption.LEFT: return 1
    elif road_option == RoadOption.RIGHT: return 2
    elif road_option == RoadOption.STRAIGHT: return 3
    else: return 0 

def cleanup_actors(world):
    actors = world.get_actors()
    for actor in actors.filter('sensor.*'):
        if actor.is_listening:
            actor.stop()
        actor.destroy()
    for actor in actors.filter('vehicle.*'):
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

def save_image(image, control, command, tl_state, episode_path, buffer_controls):
    image.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1] 
    pil_image = Image.fromarray(array)
    filename = f"{image.frame}.png"
    pil_image.save(os.path.join(episode_path, filename))
    buffer_controls.append([filename, control.steer, control.throttle, control.brake, command, tl_state])

def save_csv(episode_path, buffer_controls):
    if not buffer_controls:
        try: os.rmdir(episode_path)
        except: pass
        return
    csv_path = os.path.join(episode_path, "controls_nav.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "steer", "throttle", "brake", "command", "traffic_light"])
        writer.writerows(buffer_controls)
    print(f" Salvat: {os.path.basename(episode_path)} | {len(buffer_controls)} imagini.")

def spawn_traffic(world, client, num_vehicles=25):
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_global_distance_to_leading_vehicle(1.5)
    traffic_manager.global_percentage_speed_difference(-20)
    traffic_manager.set_synchronous_mode(False)
    
    spawned_vehicles = []
    random.shuffle(spawn_points)
    
    print(f"Spawn {num_vehicles} masini de trafic...")
    for i in range(min(num_vehicles, len(spawn_points))):
        bp = random.choice(blueprint_library.filter('vehicle.*'))
        
        if int(bp.get_attribute('number_of_wheels')) == 4:
            npc = world.try_spawn_actor(bp, spawn_points[i])
            if npc is not None:
                npc.set_autopilot(True)
                
                traffic_manager.ignore_lights_percentage(npc, 30)
                traffic_manager.ignore_walkers_percentage(npc, 0)
                traffic_manager.vehicle_percentage_speed_difference(npc, random.uniform(-30, 10))
                traffic_manager.distance_to_leading_vehicle(npc, random.uniform(1.0, 3.0))
                
                spawned_vehicles.append(npc)
                
    print(f"{len(spawned_vehicles)} vehicule in trafic.")
    return spawned_vehicles

def main():
    pygame.init()
    display = pygame.display.set_mode((450, 460))
    pygame.display.set_caption("Colectare cu Trafic + Vreme + Semafor | SPACE=Rec | R=Respawn")
    font = pygame.font.SysFont("Arial", 18)

    print("Se conecteaza la simulator...")
    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0) 
    world = client.get_world()

    current_map_name = world.get_map().name
    if not current_map_name.endswith('Town01'):
        world = client.load_world('Town01')

    cleanup_actors(world)
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    spawn_traffic(world, client, num_vehicles=25)
    time.sleep(2.0) 

    #vreme defaukt
    current_weather_idx = 0
    world.set_weather(WEATHER_PRESETS[WEATHER_NAMES[current_weather_idx]])
    print(f"[VREME] {WEATHER_NAMES[current_weather_idx]}")

    vehicle_bp = blueprint_library.filter("model3")[0]
    spawn_point = random.choice(spawn_points)
    
    vehicle = None
    while vehicle is None:
        spawn_point = random.choice(spawn_points)
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    time.sleep(1.0)

    agent = BasicAgent(vehicle, target_speed=20)
    agent.ignore_stop_signs(active=True)
    agent._base_tlight_threshold = 2.75    # stop closer to tl
    agent._base_vehicle_threshold = 3.0   # stop closer to cars
    
    current_wp = world.get_map().get_waypoint(vehicle.get_location())
    next_wps = current_wp.next(100.0)
    if next_wps: agent.set_destination(next_wps[0].transform.location)
    else: agent.set_destination(random.choice(spawn_points).location)

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
    n_pressed_last_frame = False
    manual_steer = 0.0 

    try:
        while True:
            clock.tick(60)
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            
            if keys[K_ESCAPE]: break

            #autopilot/manual M
            if keys[K_m] and not m_pressed_last_frame:
                autopilot_enabled = not autopilot_enabled
                print(f"\n>>> MOD CONDUS: {'AUTOPILOT' if autopilot_enabled else 'MANUAL (WASD)'} <<<")
            m_pressed_last_frame = keys[K_m]

            #change weather N
            if keys[K_n] and not n_pressed_last_frame:
                current_weather_idx = (current_weather_idx + 1) % len(WEATHER_NAMES)
                weather_name = WEATHER_NAMES[current_weather_idx]
                world.set_weather(WEATHER_PRESETS[weather_name])
                print(f"\n[VREME] Schimbat -> {weather_name}")
            n_pressed_last_frame = keys[K_n]

            # Respawn R
            if keys[K_r]:
                
                if is_recording and buffer_controls:
                    save_csv(current_episode_path, buffer_controls)
                    is_recording = False

                if camera.is_listening:
                    camera.stop()
                cleanup_actors(world)
                time.sleep(1.0)

                spawn_traffic(world, client, num_vehicles=25)
                time.sleep(2.0)

                vehicle = None
                while vehicle is None:
                    spawn_point = random.choice(spawn_points)
                    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
                time.sleep(1.0)

                agent = BasicAgent(vehicle, target_speed=20)
                agent.ignore_stop_signs(active=True)
                agent._base_tlight_threshold = 2.0
                agent._base_vehicle_threshold = 3.0
                current_wp = world.get_map().get_waypoint(vehicle.get_location())
                next_wps = current_wp.next(100.0)
                if next_wps:
                    agent.set_destination(next_wps[0].transform.location)
                else:
                    agent.set_destination(random.choice(spawn_points).location)

                camera = world.spawn_actor(camera_bp, cam_transform, attach_to=vehicle)
                while not image_queue.empty():
                    image_queue.get_nowait()
                camera.listen(image_queue.put)
                manual_steer = 0.0
                print(f"[CARLA] Vehicul respawnat.")
                continue

            if agent.done():
                current_loc = vehicle.get_location()
                new_dest = random.choice(spawn_points)
                while new_dest.location.distance(current_loc) < 80.0:
                    new_dest = random.choice(spawn_points)
                agent.set_destination(new_dest.location)
            
            auto_control = agent.run_step()
            current_road_option = agent.get_local_planner().target_road_option
            current_command = map_command(current_road_option)

            # adaptive speed
            if current_command == 0:      # LANE
                 agent.set_target_speed(30)
            elif current_command == 2:    # RIGHT
                agent.set_target_speed(13)
            elif current_command == 1:    # LEFT
                agent.set_target_speed(15)
            else:                         # STRAIGHT
                agent.set_target_speed(20)

            if autopilot_enabled:
                control_to_apply = auto_control
                manual_steer = auto_control.steer 
            else:
                control_to_apply = carla.VehicleControl()
                if keys[K_w]: control_to_apply.throttle = 0.5
                elif keys[K_s]: control_to_apply.brake = 1.0
                else: control_to_apply.throttle = 0.0
                
                steer_speed = 0.05
                if keys[K_a]: manual_steer = max(-1.0, manual_steer - steer_speed)
                elif keys[K_d]: manual_steer = min(1.0, manual_steer + steer_speed)
                else:
                    if manual_steer > 0: manual_steer = max(0.0, manual_steer - steer_speed)
                    elif manual_steer < 0: manual_steer = min(0.0, manual_steer + steer_speed)
                control_to_apply.steer = manual_steer

            vehicle.apply_control(control_to_apply)

# =====================================================================
#                        Traffic Lights
# =====================================================================
            # Method 1: API
            tl = vehicle.get_traffic_light()
            if tl is not None:
                tl_loc = tl.get_location()
                v_loc = vehicle.get_location()
                v_fwd = vehicle.get_transform().get_forward_vector()
                dx = tl_loc.x - v_loc.x
                dy = tl_loc.y - v_loc.y
                dot = dx * v_fwd.x + dy * v_fwd.y
                
                if dot < 0:
                    current_tl_state = 0
                else:
                    tl_state = tl.get_state()
                    if tl_state == carla.TrafficLightState.Red: current_tl_state = 1
                    elif tl_state == carla.TrafficLightState.Yellow: current_tl_state = 2
                    else: current_tl_state = 0
            else:
                # Method 2: search for TL
                v_loc = vehicle.get_location()
                v_fwd = vehicle.get_transform().get_forward_vector()
                v_wp = world.get_map().get_waypoint(v_loc)
                current_tl_state = 0
                best_dist = 999.0
                
                for tl_actor in world.get_actors().filter('traffic.traffic_light*'):
                    tl_loc = tl_actor.get_location()
                    dist = v_loc.distance(tl_loc)
                    if dist > 30.0: continue
                    dx = tl_loc.x - v_loc.x
                    dy = tl_loc.y - v_loc.y
                    dot = dx * v_fwd.x + dy * v_fwd.y
                    if dot < 0: continue
                    tl_wp = world.get_map().get_waypoint(tl_loc)
                    if tl_wp.road_id != v_wp.road_id: continue
                    if (tl_wp.lane_id * v_wp.lane_id) < 0: continue  # contrasens
                    if dist < best_dist:
                        best_dist = dist
                        state = tl_actor.get_state()
                        if state == carla.TrafficLightState.Red: current_tl_state = 1
                        elif state == carla.TrafficLightState.Yellow: current_tl_state = 2
                        else: current_tl_state = 0

            if keys[K_SPACE] and not is_recording:
                is_recording = True
                current_episode_path = get_next_episode_path(ROOT_SAVE_FOLDER)
                os.makedirs(current_episode_path, exist_ok=True)
                buffer_controls = []
                time.sleep(0.2) 
            elif not keys[K_SPACE] and is_recording:
                is_recording = False
                save_csv(current_episode_path, buffer_controls)

# =====================================================================
#                        DISPLAY PYGAME
# =====================================================================

            bg_color = (200, 0, 0) if is_recording else (0, 0, 0)
            display.fill(bg_color)
            
            driver_str = "AUTO" if autopilot_enabled else "MANUAL"
            cmd_str = ["LANE", "LEFT", "RIGHT", "STRAIGHT"][current_command]
            weather_str = WEATHER_NAMES[current_weather_idx]
            
            text_1 = font.render(f"Driver: {driver_str} | REC: {'DA' if is_recording else 'NU'}", True, (255,255,255))
            text_2 = font.render(f"GPS: {cmd_str} | S: {control_to_apply.steer:.2f} | T: {control_to_apply.throttle:.2f} | B: {control_to_apply.brake:.2f}", True, (255,255,255))
            
            tl_labels = ["VERDE/NIMIC", "ROSU", "GALBEN"]
            tl_colors_display = [(0, 255, 0), (255, 0, 0), (255, 255, 0)]
            text_tl = font.render(f"SEMAFOR: {tl_labels[current_tl_state]}", True, tl_colors_display[current_tl_state])
            
            text_3 = font.render(f"[M] Manual | [SPACE] Rec | [R] Respawn | [N] Vreme", True, (150,150,150))
            
            #colors
            weather_colors = {
                "ZI_SENINA": (255, 255, 0),
                "INNORAT": (180, 180, 180),
                "PLOAIE_USOARA": (100, 150, 255),
                "PLOAIE_PUTERNICA": (50, 80, 200),
                "CEATA": (200, 200, 200),
                "APUS": (255, 150, 50),
            }
            w_color = weather_colors.get(weather_str, (255, 255, 255))
            text_4 = font.render(f"VREME: {weather_str} | [N] Schimba", True, w_color)
            
            text_5 = font.render(f"RADAR GPS (2D) \/", True, (255, 255, 0))
            
            display.blit(text_1, (10, 10))
            display.blit(text_2, (10, 40))
            display.blit(text_tl, (10, 70))
            display.blit(text_3, (10, 100))
            display.blit(text_4, (10, 130))
            display.blit(text_5, (155, 165))

            if vehicle.is_alive:
                v_transform = vehicle.get_transform()
                v_x = v_transform.location.x
                v_y = v_transform.location.y
                v_yaw = math.radians(v_transform.rotation.yaw)
                
                route_trace = list(agent.get_local_planner()._waypoints_queue)
                radar_center_x, radar_center_y = 225, 390
                pygame.draw.circle(display, (0, 150, 255), (radar_center_x, radar_center_y), 6)

                for wp, _ in route_trace:
                    w_x = wp.transform.location.x
                    w_y = wp.transform.location.y
                    dx = w_x - v_x
                    dy = w_y - v_y
                    
                    dist = math.sqrt(dx**2 + dy**2)
                    rel_x = dx * math.cos(v_yaw) + dy * math.sin(v_yaw)
                    rel_y = -dx * math.sin(v_yaw) + dy * math.cos(v_yaw)
                    
                    if dist < 40.0 and rel_x > -2.0:
                        scale = 4  
                        screen_x = int(radar_center_x + rel_y * scale)
                        screen_y = int(radar_center_y - rel_x * scale) 
                        if 0 <= screen_x <= 450 and 0 <= screen_y <= 460:
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
                    save_image(last_image, control_to_apply, current_command, current_tl_state, current_episode_path, buffer_controls)
            except queue.Empty: pass

    finally:
        print("\n[OPRIRE] Se opresc senzorii si se curata traficul...")
        if is_recording and buffer_controls:
            save_csv(current_episode_path, buffer_controls)
        if 'camera' in locals() and camera is not None:
            if camera.is_listening:
                camera.stop()
        cleanup_actors(world) 
        pygame.quit()
        print("Exit.")

if __name__ == "__main__":
    main()