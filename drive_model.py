import carla
import time
import os
import numpy as np
from PIL import Image
import queue
import pygame
from pygame.locals import K_ESCAPE, K_v, K_r, K_m, K_1, K_2, K_3, K_4, K_5
import random
import sys
import glob
import math
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

try:
    sys.path.append(glob.glob('../carla')[0])
except IndexError:
    pass

from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import RoadOption


MODEL_PATH = "model_nav_traffic.pth" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEERING_HISTORY_SIZE = 2 

# --- HOOK PENTRU FEATURE MAPS ---
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


CONV_LAYER_INDICES = {1: 0, 2: 2, 3: 4, 4: 6, 5: 8}
CONV_LAYER_NAMES = {1: 'conv1', 2: 'conv2', 3: 'conv3', 4: 'conv4', 5: 'conv5'}


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
    time.sleep(0.5)


def spawn_traffic(world, client, num_vehicles=30):
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_global_distance_to_leading_vehicle(2.0)
    spawned_vehicles = []
    random.shuffle(spawn_points)
    for i in range(min(num_vehicles, len(spawn_points))):
        bp = random.choice(blueprint_library.filter('vehicle.*'))
        if int(bp.get_attribute('number_of_wheels')) == 4:
            npc = world.try_spawn_actor(bp, spawn_points[i])
            if npc is not None:
                npc.set_autopilot(True)
                spawned_vehicles.append(npc)
    print(f"[TRAFIC] {len(spawned_vehicles)} vehicule npc.")
    return spawned_vehicles


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
        self.command_fc = nn.Sequential(nn.Linear(4, 16), nn.ReLU())
        self.joint_fc = nn.Sequential(
            nn.Linear(1152 + 16, 256), nn.ReLU(),
            nn.Dropout(p=0.3), 
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(p=0.2), 
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 3) 
        )

    def forward(self, img, cmd):
        img_feats = self.conv_layers(img)
        cmd_onehot = F.one_hot(cmd.long(), num_classes=4).float()
        cmd_features = self.command_fc(cmd_onehot)
        combined = torch.cat((img_feats, cmd_features), dim=1)
        return self.joint_fc(combined)

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
    image.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1] 
    pil_image = Image.fromarray(array)
    return transform_pipeline(pil_image).unsqueeze(0).to(DEVICE)

def carla_image_to_rgb(image):
    """Extrage imaginea RGB din cadrul CARLA (pentru afișare în heatmap)."""
    image.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
    pil_image = Image.fromarray(array)
    pil_cropped = pil_image.crop((0, 80, 320, 240))
    pil_resized = pil_cropped.resize((200, 66))
    return np.array(pil_resized)


def main():
    pygame.init()
    display = pygame.display.set_mode((450, 400))
    pygame.display.set_caption("AI Driving | Trafic Inclus")
    font = pygame.font.SysFont("Arial", 18)

    print("Se conecteaza la simulator...")
    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0) 
    world = client.get_world()

    current_map_name = world.get_map().name
    if not current_map_name.endswith('Town01'):
        world = client.load_world('Town01')
        time.sleep(2.0)

    model = ConditionalNvidiaModel().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
            model.eval()
            print(f"\n[AI] Model {MODEL_PATH} încărcat pe {DEVICE}!")
        except Exception as e:
            print(f"\n[EROARE] Eroare la incarcare: {e}")
            return
    else:
        print(f"\n[EROARE] Nu am gasit fisierul {MODEL_PATH}!")
        return

    #hook-uri pe TOATE cele 5 straturi conv
    for layer_num, layer_idx in CONV_LAYER_INDICES.items():
        layer_name = CONV_LAYER_NAMES[layer_num]
        model.conv_layers[layer_idx].register_forward_hook(get_activation(layer_name))

    cleanup_actors(world)
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    spectator = world.get_spectator()
    clock = pygame.time.Clock()

    
    spawn_traffic(world, client, num_vehicles=30)
    time.sleep(2.0)

    
    vehicle_bp = blueprint_library.filter("model3")[0]
    vehicle = None
    while vehicle is None:
        spawn_point = random.choice(spawn_points)
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    time.sleep(1.0)

    
    agent = BasicAgent(vehicle, target_speed=20)
    agent.ignore_traffic_lights(active=True)
    agent.ignore_stop_signs(active=True)

    current_wp = world.get_map().get_waypoint(vehicle.get_location())
    next_wps = current_wp.next(100.0)
    if next_wps:
        agent.set_destination(next_wps[0].transform.location)
    else:
        agent.set_destination(random.choice(spawn_points).location)

   
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", "320")
    camera_bp.set_attribute("image_size_y", "240")
    camera_bp.set_attribute("fov", "90")
    camera_bp.set_attribute("sensor_tick", "0.1")
    cam_transform = carla.Transform(carla.Location(x=1.5, z=1.4), carla.Rotation(pitch=-15.0))
    camera = world.spawn_actor(camera_bp, cam_transform, attach_to=vehicle)

    image_queue = queue.Queue()
    camera.listen(image_queue.put)

    steering_history = [0.0] * STEERING_HISTORY_SIZE
    follow_mode = True
    current_command = 0

    # --- HEATMAP STATE ---
    show_heatmap = False
    m_pressed_last_frame = False
    heatmap_fig = None
    heatmap_ax = None
    heatmap_im = None
    last_raw_image = None
    active_layer = 1  # stratul activ (1-5)

    try:
        while True:
            clock.tick(60)
            pygame.event.pump()
            keys = pygame.key.get_pressed()

            if keys[K_ESCAPE]:
                break
            if keys[K_v]:
                follow_mode = not follow_mode
                time.sleep(0.3)

            # Toggle M
            if keys[K_m] and not m_pressed_last_frame:
                show_heatmap = not show_heatmap
                if show_heatmap:
                    plt.ion()
                    heatmap_fig, heatmap_ax = plt.subplots(1, 1, figsize=(8, 3))
                    heatmap_fig.canvas.manager.set_window_title("LIVE Heatmap")
                    heatmap_ax.set_title(f"Live heatmap (Conv{active_layer})", fontsize=12)
                    heatmap_ax.axis('off')
                    dummy = np.zeros((66, 200, 3), dtype=np.uint8)
                    heatmap_im = heatmap_ax.imshow(dummy)
                    plt.tight_layout()
                    plt.show(block=False)
                    print(f"[HEATMAP] Fereastra deschisa — Conv{active_layer}. Taste 1-5 pentru a schimba stratul.")
                else:
                    if heatmap_fig is not None:
                        plt.close(heatmap_fig)
                        heatmap_fig = None
                        heatmap_ax = None
                        heatmap_im = None
                    print("[HEATMAP] Fereastra inchisa.")
            m_pressed_last_frame = keys[K_m]

            #change layer 1-5
            for key, layer_num in [(K_1, 1), (K_2, 2), (K_3, 3), (K_4, 4), (K_5, 5)]:
                if keys[key] and active_layer != layer_num:
                    active_layer = layer_num
                    if show_heatmap and heatmap_ax is not None:
                        heatmap_ax.set_title(f"LIVE Heatmap (Conv{active_layer})", fontsize=12)
                    print(f"[HEATMAP] Strat schimbat -> Conv{active_layer}")
                    time.sleep(0.2)
                    break

            if keys[K_r]:
                if show_heatmap and heatmap_fig is not None:
                    plt.close(heatmap_fig)
                    heatmap_fig = None
                    show_heatmap = False

                if camera.is_listening:
                    camera.stop()
                cleanup_actors(world)
                time.sleep(1.0)

                spawn_traffic(world, client, num_vehicles=30)
                time.sleep(2.0)

                vehicle = None
                while vehicle is None:
                    spawn_point = random.choice(spawn_points)
                    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
                time.sleep(1.0)

                agent = BasicAgent(vehicle, target_speed=20)
                agent.ignore_traffic_lights(active=True)
                agent.ignore_stop_signs(active=True)
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
                steering_history = [0.0] * STEERING_HISTORY_SIZE
                print(f"[CARLA] Vehicul respawnat.")
                continue

            
            if agent.done():
                current_loc = vehicle.get_location()
                new_dest = random.choice(spawn_points)
                while new_dest.location.distance(current_loc) < 80.0:
                    new_dest = random.choice(spawn_points)
                agent.set_destination(new_dest.location)
                print(f"[GPS] Traseu nou setat!")

            
            auto_control = agent.run_step()
            current_road_option = agent.get_local_planner().target_road_option
            current_command = map_command(current_road_option)

            
            control_to_apply = carla.VehicleControl()

            try:
                last_image = None
                while not image_queue.empty():
                    last_image = image_queue.get_nowait()

                if last_image is not None:
                    last_raw_image = last_image
                    img_t = image_to_tensor(last_image)
                    cmd_t = torch.tensor([current_command], dtype=torch.long).to(DEVICE)

                    with torch.no_grad():
                        predictions = model(img_t, cmd_t)[0]
                        raw_steer = float(predictions[0])
                        raw_throttle = float(predictions[1])
                        raw_brake = float(predictions[2])

                    steering_history.pop(0)
                    steering_history.append(raw_steer)
                    avg_steer = sum(steering_history) / len(steering_history)
                    control_to_apply.steer = avg_steer

                    vel = vehicle.get_velocity()
                    current_speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

                    if raw_brake > 0.2:
                        control_to_apply.brake = max(0.4, min(1.0, raw_brake * 2.0))
                        control_to_apply.throttle = 0.0
                    else:
                        control_to_apply.brake = 0.0
                        target_speed = raw_throttle * 40.0
                        speed_error = target_speed - current_speed
                        if speed_error > 0:
                            calc_throttle = speed_error * 0.15
                            control_to_apply.throttle = max(0.25, min(0.75, calc_throttle))
                        else:
                            control_to_apply.throttle = 0.0

                    vehicle.apply_control(control_to_apply)

                    # --- ACTUALIZAARE HEATMAP LIVE ---
                    active_layer_name = CONV_LAYER_NAMES[active_layer]
                    if show_heatmap and heatmap_fig is not None and active_layer_name in activation and last_raw_image is not None:
                        try:
                            rgb_frame = carla_image_to_rgb(last_raw_image)

                            act = activation[active_layer_name].squeeze().cpu().numpy()
                            mean_activation = np.mean(act, axis=0)

                            mean_activation = mean_activation - mean_activation.min()
                            if mean_activation.max() > 0:
                                mean_activation = mean_activation / mean_activation.max()

                            act_pil = Image.fromarray((mean_activation * 255).astype(np.uint8))
                            act_resized = act_pil.resize((200, 66), Image.BILINEAR)
                            act_np = np.array(act_resized).astype(np.float32) / 255.0

                            heatmap_colored = plt.cm.jet(act_np)[:, :, :3]

                            rgb_normalized = rgb_frame.astype(np.float32) / 255.0
                            alpha = 0.5
                            overlay = (1 - alpha) * rgb_normalized + alpha * heatmap_colored
                            overlay = np.clip(overlay, 0, 1)

                            heatmap_im.set_data(overlay)
                            heatmap_fig.canvas.draw_idle()
                            heatmap_fig.canvas.flush_events()
                        except Exception:
                            pass

            except queue.Empty:
                pass

            # --- DISPLAY PYGAME ---
            display.fill((0, 0, 0))
            cmd_str = ["LANE", "LEFT", "RIGHT", "STRAIGHT"][current_command]

            text_1 = font.render(f"Driver: AI | Model: {MODEL_PATH}", True, (255,255,255))
            text_2 = font.render(f"GPS: {cmd_str} | Steer: {control_to_apply.steer:.2f} | T: {control_to_apply.throttle:.2f} | B: {control_to_apply.brake:.2f}", True, (255,255,255))
            text_3 = font.render(f"[V] Camera | [R] Respawn | [M] Heatmap | [ESC] Exit", True, (150,150,150))
            text_4 = font.render(f"RADAR GPS (2D) \/", True, (255, 255, 0))
            
            if show_heatmap:
                heatmap_status = f"HEATMAP: ON | Conv{active_layer} | [1-5] Schimba strat"
                color_status = (0, 255, 0)
            else:
                heatmap_status = "HEATMAP: OFF | [M] Deschide"
                color_status = (150, 150, 150)
            text_5 = font.render(heatmap_status, True, color_status)

            display.blit(text_1, (10, 10))
            display.blit(text_2, (10, 40))
            display.blit(text_3, (10, 70))
            display.blit(text_5, (10, 100))
            display.blit(text_4, (155, 130))

            if vehicle.is_alive:
                v_transform = vehicle.get_transform()
                v_x = v_transform.location.x
                v_y = v_transform.location.y
                v_yaw = math.radians(v_transform.rotation.yaw)

                route_trace = list(agent.get_local_planner()._waypoints_queue)

                radar_center_x, radar_center_y = 225, 350
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
                        if 0 <= screen_x <= 450 and 0 <= screen_y <= 400:
                            pygame.draw.circle(display, (0, 255, 0), (screen_x, screen_y), 3)

            pygame.display.flip()

            if follow_mode and vehicle.is_alive:
                t = vehicle.get_transform()
                spectator.set_transform(carla.Transform(
                    t.location - 6 * t.get_forward_vector() + carla.Location(z=3.0),
                    carla.Rotation(pitch=-15.0, yaw=t.rotation.yaw)
                ))

            if show_heatmap and heatmap_fig is not None:
                try:
                    heatmap_fig.canvas.flush_events()
                except Exception:
                    pass

    finally:
        print("\n[OPRIRE] Se opresc senzorii și conexiunea...")
        if show_heatmap and heatmap_fig is not None:
            plt.close(heatmap_fig)
        if camera is not None and camera.is_listening:
            camera.stop()
        cleanup_actors(world)
        pygame.quit()
        print("Exit.")

if __name__ == "__main__":
    main()