import carla
import time
import os
import numpy as np
from PIL import Image
import queue
import pygame
from pygame.locals import K_ESCAPE, K_v, K_r, K_m, K_n, K_t, K_1, K_2, K_3, K_4
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


MODEL_PATH = "model_nav_traffic_vit.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEERING_HISTORY_SIZE = 2
BRAKE_HISTORY_SIZE = 5        # faster reaction for breaking

# === Smart steering system based on speeed ===

SPEED_STEER_FREE = 15.0       # km/h — sub viteza asta, zero dampening  
SPEED_STEER_MAX = 30.0        # km/h — la viteza asta, dampening maxim
SPEED_STEER_MIN_FACTOR = 0.55 # factorul minim la viteza mare
MAX_STEER_CHANGE = 0.05       # cat de mult se poate schimba steering-ul per frame

# === WEATHER PRSETS ===
WEATHER_PRESETS = {
    "ZI_SENINA": carla.WeatherParameters(
        sun_altitude_angle=70.0, cloudiness=10.0, precipitation=0.0,
        precipitation_deposits=0.0, wind_intensity=10.0,
        fog_density=0.0, fog_distance=0.0, wetness=0.0, sun_azimuth_angle=0.0
    ),
    "INNORAT": carla.WeatherParameters(
        sun_altitude_angle=50.0, cloudiness=80.0, precipitation=0.0,
        precipitation_deposits=0.0, wind_intensity=30.0,
        fog_density=0.0, fog_distance=0.0, wetness=0.0, sun_azimuth_angle=90.0
    ),
    "PLOAIE_USOARA": carla.WeatherParameters(
        sun_altitude_angle=40.0, cloudiness=70.0, precipitation=30.0,
        precipitation_deposits=30.0, wind_intensity=40.0,
        fog_density=5.0, fog_distance=0.0, wetness=40.0, sun_azimuth_angle=180.0
    ),
    "PLOAIE_PUTERNICA": carla.WeatherParameters(
        sun_altitude_angle=30.0, cloudiness=90.0, precipitation=70.0,
        precipitation_deposits=70.0, wind_intensity=70.0,
        fog_density=10.0, fog_distance=0.0, wetness=80.0, sun_azimuth_angle=270.0
    ),
    "CEATA": carla.WeatherParameters(
        sun_altitude_angle=45.0, cloudiness=50.0, precipitation=0.0,
        precipitation_deposits=0.0, wind_intensity=5.0,
        fog_density=40.0, fog_distance=30.0, wetness=20.0, sun_azimuth_angle=45.0
    ),
    "APUS": carla.WeatherParameters(
        sun_altitude_angle=10.0, cloudiness=20.0, precipitation=0.0,
        precipitation_deposits=0.0, wind_intensity=10.0,
        fog_density=5.0, fog_distance=0.0, wetness=0.0, sun_azimuth_angle=220.0
    ),
}
WEATHER_NAMES = list(WEATHER_PRESETS.keys())


# === HYBRID VIT CONV STEM

class ConvStem(nn.Module):
    """
    Mini-CNN care inlocuieste patch embedding-ul brut.
    3x66x200 -> 48x33x100 -> 96x17x50 -> 128x9x25
    Rezultat: 225 tokens cu features locale extrase.
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
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        return x


class TransformerBlock(nn.Module):
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
    Hybrid ViT cu Conv Stem + GPS ca Token + Semafor ca Token.
    Conv Stem: 3 straturi CNN -> 225 patches cu features locale
    Secventa: [CLS] [GPS] [TL] [patch_1] ... [patch_225] = 228 tokens
    """
    def __init__(self, img_h=66, img_w=200,
                 embed_dim=128, num_heads=4, num_layers=4, mlp_ratio=4, dropout=0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.conv_stem = ConvStem(embed_dim)
        
        self.num_patches_h = math.ceil(img_h / 8)   # 9
        self.num_patches_w = math.ceil(img_w / 8)    # 25
        self.num_patches = self.num_patches_h * self.num_patches_w  # 225

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        self.gps_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.tl_embed = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 3, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, img, cmd, tl):
        B = img.shape[0]
        x = self.conv_stem(img)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        cmd_onehot = F.one_hot(cmd.long(), num_classes=4).float()
        gps_token = self.gps_embed(cmd_onehot).unsqueeze(1)
        tl_onehot = F.one_hot(tl.long(), num_classes=3).float()
        tl_token = self.tl_embed(tl_onehot).unsqueeze(1)
        x = torch.cat([cls, gps_token, tl_token, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head(cls_out)

    def get_attention_maps(self, layer_idx=-1):
        """Atentia CLS -> patches."""
        block = self.blocks[layer_idx]
        if block.attn_weights is None:
            return None
        cls_attn = block.attn_weights[0, 0, 3:]  # skip CLS(0), GPS(1), TL(2)
        attn_map = cls_attn.reshape(self.num_patches_h, self.num_patches_w)
        attn_map = attn_map - attn_map.min()
        if attn_map.max() > 0:
            attn_map = attn_map / attn_map.max()
        return attn_map.detach().cpu().numpy()


# === UTILITARE ===

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

def crop_img(img):
    return img.crop((0, 40, 320, 240))

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
    image.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
    pil_image = Image.fromarray(array)
    pil_cropped = pil_image.crop((0, 40, 320, 240))
    pil_resized = pil_cropped.resize((200, 66))
    return np.array(pil_resized)


# === MAIN ===

def main():
    pygame.init()
    display = pygame.display.set_mode((450, 550))
    pygame.display.set_caption("AI Driving Hybrid ViT (Conv Stem + GPS Token)")
    font = pygame.font.SysFont("Arial", 18)

    print("Se conecteaza la simulator...")
    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    world = client.get_world()

    current_map_name = world.get_map().name
    if not current_map_name.endswith('Town01'):
        world = client.load_world('Town01')
        time.sleep(2.0)

    model = ConditionalViTModel().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
            model.eval()
            print(f"\n[AI] Model Hybrid ViT {MODEL_PATH} incarcat pe {DEVICE}!")
            print(f"     Conv Stem: 3->48->96->128 | {model.num_patches} tokens + CLS + GPS")
        except Exception as e:
            print(f"\n[EROARE] Eroare la incarcare: {e}")
            return
    else:
        print(f"\n[EROARE] Nu am gasit fisierul {MODEL_PATH}!")
        return

    cleanup_actors(world)
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    spectator = world.get_spectator()
    clock = pygame.time.Clock()

    spawn_traffic(world, client, num_vehicles=30)
    time.sleep(2.0)

    # default weather
    current_weather_idx = 0
    n_pressed_last_frame = False
    world.set_weather(WEATHER_PRESETS[WEATHER_NAMES[current_weather_idx]])
    print(f"[VREME] {WEATHER_NAMES[current_weather_idx]}")

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
    brake_history = [0.0] * BRAKE_HISTORY_SIZE
    follow_mode = True
    current_command = 0

    # --- ATTENTION MAP STATE ---
    show_attn = False
    m_pressed_last_frame = False
    attn_fig = None
    attn_ax = None
    attn_im = None
    last_raw_image = None
    active_layer = 4
    smooth_attn = None

    # --- SMOOTH CAMERA ---
    CAMERA_SMOOTH = 0.1
    smooth_cam_x = None
    smooth_cam_y = None
    smooth_cam_z = None
    smooth_cam_yaw = None

    
    last_control = carla.VehicleControl()
    display_speed = 0.0
    display_speed_factor = 1.0

    # --- SAFETY NET TOGGLE ---
    safety_net_enabled = True
    t_pressed_last_frame = False

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

            # weather change N
            if keys[K_n] and not n_pressed_last_frame:
                current_weather_idx = (current_weather_idx + 1) % len(WEATHER_NAMES)
                weather_name = WEATHER_NAMES[current_weather_idx]
                world.set_weather(WEATHER_PRESETS[weather_name])
                print(f"[VREME] Schimbat -> {weather_name}")
            n_pressed_last_frame = keys[K_n]

            # Toggle T — safety net tl
            if keys[K_t] and not t_pressed_last_frame:
                safety_net_enabled = not safety_net_enabled
                status = "ACTIV" if safety_net_enabled else "DEZACTIVAT"
                print(f"[SAFETY NET] {status}")
            t_pressed_last_frame = keys[K_t]

            # Toggle M — attention map
            if keys[K_m] and not m_pressed_last_frame:
                show_attn = not show_attn
                if show_attn:
                    plt.ion()
                    attn_fig, attn_ax = plt.subplots(1, 1, figsize=(8, 3))
                    attn_fig.canvas.manager.set_window_title("LIVE Attention Map (Hybrid ViT)")
                    attn_ax.set_title(f"CLS Attention — Layer {active_layer}", fontsize=12)
                    attn_ax.axis('off')
                    dummy = np.zeros((66, 200, 3), dtype=np.uint8)
                    attn_im = attn_ax.imshow(dummy)
                    plt.tight_layout()
                    plt.show(block=False)
                    smooth_attn = None
                    print(f"[ATTN] Fereastra deschisa — Layer {active_layer}. [1-4] Schimba strat")
                else:
                    if attn_fig is not None:
                        plt.close(attn_fig)
                        attn_fig = None
                        attn_ax = None
                        attn_im = None
                    smooth_attn = None
                    print("[ATTN] Fereastra inchisa.")
            m_pressed_last_frame = keys[K_m]

            # change layer 1-4
            for key, layer_num in [(K_1, 1), (K_2, 2), (K_3, 3), (K_4, 4)]:
                if keys[key] and active_layer != layer_num:
                    active_layer = layer_num
                    smooth_attn = None
                    if show_attn and attn_ax is not None:
                        attn_ax.set_title(f"CLS Attention — Layer {active_layer}", fontsize=12)
                    print(f"[ATTN] Strat schimbat -> Layer {active_layer}")
                    time.sleep(0.2)
                    break

            # Respawn R
            if keys[K_r]:
                if show_attn and attn_fig is not None:
                    plt.close(attn_fig)
                    attn_fig = None
                    show_attn = False
                    smooth_attn = None

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
                brake_history = [0.0] * BRAKE_HISTORY_SIZE
                last_control = carla.VehicleControl()
                smooth_cam_x = None
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

            # === READ TL STATE from API ===
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

            try:
                last_image = None
                while not image_queue.empty():
                    last_image = image_queue.get_nowait()

                if last_image is not None:
                    last_raw_image = last_image
                    img_t = image_to_tensor(last_image)
                    cmd_t = torch.tensor([current_command], dtype=torch.long).to(DEVICE)
                    tl_t = torch.tensor([current_tl_state], dtype=torch.long).to(DEVICE)

                    with torch.no_grad():
                        predictions = model(img_t, cmd_t, tl_t)[0]
                        raw_steer = float(predictions[0])
                        raw_throttle = float(predictions[1])
                        raw_brake = float(predictions[2])

                    steering_history.pop(0)
                    steering_history.append(raw_steer)
                    avg_steer = sum(steering_history) / len(steering_history)

                    vel = vehicle.get_velocity()
                    current_speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                    
                    # === SMART STERING BASED ON SPED ===
                    if current_speed < SPEED_STEER_FREE:
                        speed_factor = 1.0
                    elif current_speed > SPEED_STEER_MAX:
                        speed_factor = SPEED_STEER_MIN_FACTOR
                    else:
                        t = (current_speed - SPEED_STEER_FREE) / (SPEED_STEER_MAX - SPEED_STEER_FREE)
                        speed_factor = 1.0 - t * (1.0 - SPEED_STEER_MIN_FACTOR)
                    
                    # Rate limiter
                    desired_steer = max(-1.0, min(1.0, avg_steer * speed_factor))
                    steer_diff = desired_steer - last_control.steer
                    if abs(steer_diff) > MAX_STEER_CHANGE:
                        desired_steer = last_control.steer + MAX_STEER_CHANGE * (1 if steer_diff > 0 else -1)
                    last_control.steer = desired_steer
                    
                    display_speed = current_speed
                    display_speed_factor = speed_factor

                    # === BRAKE GRADUAL (3 zone) ===
                    brake_history.pop(0)
                    brake_history.append(raw_brake)
                    avg_brake = sum(brake_history) / len(brake_history)

                    if avg_brake > 0.30:
                        last_control.brake = max(0.4, min(1.0, avg_brake * 2.0))
                        last_control.throttle = 0.0
                    elif avg_brake > 0.08:
                        last_control.brake = avg_brake
                        last_control.throttle = 0.1
                    else:                   
                        last_control.brake = 0.0

                        target_speed = raw_throttle * 55.0
                        if current_command == 0:      # LANE
                            max_speed = 30.0
                        elif current_command == 2:    # RIGHT
                            max_speed = 13.0
                        elif current_command == 1:    # LEFT
                            max_speed = 15.0
                        else:                         # STRAIGHT
                            max_speed = 20.0
                        target_speed = min(target_speed, max_speed)
                        speed_error = target_speed - current_speed
                        if speed_error > 0:
                            calc_throttle = speed_error * 0.15
                            last_control.throttle = max(0.25, min(0.75, calc_throttle))
                        elif current_speed < 3.0:
                            last_control.throttle = 0.15
                        else:
                            last_control.throttle = 0.0

                    # === SAFETY NET TL ===
                    if safety_net_enabled and current_tl_state == 1:  # ROSU
                        last_control.brake = 0.8
                        last_control.throttle = 0.0

                    vehicle.apply_control(last_control)

                    # ---LIVE ATTENTION MAP---
                    if show_attn and attn_fig is not None and last_raw_image is not None:
                        try:
                            rgb_frame = carla_image_to_rgb(last_raw_image)
                            raw_attn = model.get_attention_maps(layer_idx=active_layer - 1)

                            if raw_attn is not None:
                                if smooth_attn is None or smooth_attn.shape != raw_attn.shape:
                                    smooth_attn = raw_attn.copy()
                                else:
                                    smooth_attn = 0.7 * smooth_attn + 0.3 * raw_attn

                                attn_pil = Image.fromarray((smooth_attn * 255).astype(np.uint8))
                                attn_resized = attn_pil.resize((200, 66), Image.BILINEAR)
                                attn_np = np.array(attn_resized).astype(np.float32) / 255.0

                                heatmap_colored = plt.cm.jet(attn_np)[:, :, :3]
                                rgb_normalized = rgb_frame.astype(np.float32) / 255.0
                                alpha = 0.5
                                overlay = (1 - alpha) * rgb_normalized + alpha * heatmap_colored
                                overlay = np.clip(overlay, 0, 1)

                                attn_im.set_data(overlay)
                                attn_fig.canvas.draw_idle()
                                attn_fig.canvas.flush_events()
                        except Exception:
                            pass

            except queue.Empty:
                pass

            # --- DISPLAY PYGAME ---
            display.fill((0, 0, 0))
            cmd_str = ["LANE", "LEFT", "RIGHT", "STRAIGHT"][current_command]

            text_1 = font.render(f"Driver: AI Hybrid ViT | {MODEL_PATH}", True, (255, 255, 255))
            text_2 = font.render(f"GPS: {cmd_str} | Steer: {last_control.steer:.2f} | T: {last_control.throttle:.2f} | B: {last_control.brake:.2f}", True, (255, 255, 255))
            text_6 = font.render(f"Speed: {display_speed:.0f} km/h | SteerFactor: {display_speed_factor:.2f}", True, (100, 200, 255))
            
            tl_labels = ["VERDE/NIMIC", "ROSU", "GALBEN"]
            tl_colors_display = [(0, 255, 0), (255, 0, 0), (255, 255, 0)]
            text_tl = font.render(f"SEMAFOR: {tl_labels[current_tl_state]}", True, tl_colors_display[current_tl_state])
            
            text_3 = font.render(f"[V] Camera | [R] Respawn | [M] Attention | [T] SafetyNet | [ESC] Exit", True, (150, 150, 150))

            if safety_net_enabled:
                sn_text = font.render("SAFETY NET: ON", True, (0, 255, 0))
            else:
                sn_text = font.render("SAFETY NET: OFF — modelul decide singur", True, (255, 80, 80))

            if show_attn:
                attn_status = f"ATTN: ON | Layer {active_layer}/4 | [1-4] Schimba strat"
                color_status = (0, 255, 0)
            else:
                attn_status = "ATTENTION: OFF | [M] Deschide"
                color_status = (150, 150, 150)
            text_5 = font.render(attn_status, True, color_status)

            weather_str = WEATHER_NAMES[current_weather_idx]
            weather_colors = {
                "ZI_SENINA": (255, 255, 0), "INNORAT": (180, 180, 180),
                "PLOAIE_USOARA": (100, 150, 255), "PLOAIE_PUTERNICA": (50, 80, 200),
                "CEATA": (200, 200, 200), "APUS": (255, 150, 50),
            }
            w_color = weather_colors.get(weather_str, (255, 255, 255))
            text_7 = font.render(f"VREME: {weather_str} | [N] Schimba", True, w_color)

            text_4 = font.render(f"RADAR GPS (2D) \/", True, (255, 255, 0))

            display.blit(text_1, (10, 10))
            display.blit(text_2, (10, 40))
            display.blit(text_6, (10, 70))
            display.blit(text_tl, (10, 100))
            display.blit(text_3, (10, 130))
            display.blit(text_5, (10, 160))
            display.blit(text_7, (10, 190))
            display.blit(sn_text, (10, 220))
            display.blit(text_4, (155, 255))

            #radar GPS 2D
            if vehicle.is_alive:
                v_transform = vehicle.get_transform()
                v_x = v_transform.location.x
                v_y = v_transform.location.y
                v_yaw = math.radians(v_transform.rotation.yaw)

                route_trace = list(agent.get_local_planner()._waypoints_queue)

                radar_center_x, radar_center_y = 225, 480
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
                        if 0 <= screen_x <= 450 and 0 <= screen_y <= 550:
                            pygame.draw.circle(display, (0, 255, 0), (screen_x, screen_y), 3)

            pygame.display.flip()

            if follow_mode and vehicle.is_alive:
                t = vehicle.get_transform()
                #pozitia tinta a camerei (in spatele masinii)
                target_loc = t.location - 6 * t.get_forward_vector() + carla.Location(z=3.0)
                target_yaw = t.rotation.yaw

                #prima iteratie: initializare directa
                if smooth_cam_x is None:
                    smooth_cam_x = target_loc.x
                    smooth_cam_y = target_loc.y
                    smooth_cam_z = target_loc.z
                    smooth_cam_yaw = target_yaw
                else:
                    #lerp (interpolare liniara) pentru miscare lina
                    smooth_cam_x += (target_loc.x - smooth_cam_x) * CAMERA_SMOOTH
                    smooth_cam_y += (target_loc.y - smooth_cam_y) * CAMERA_SMOOTH
                    smooth_cam_z += (target_loc.z - smooth_cam_z) * CAMERA_SMOOTH
                    
                    #lerp special pentru yaw (sa nu sara la trecerea 360->0)
                    yaw_diff = target_yaw - smooth_cam_yaw
                    if yaw_diff > 180: yaw_diff -= 360
                    elif yaw_diff < -180: yaw_diff += 360
                    smooth_cam_yaw += yaw_diff * CAMERA_SMOOTH

                spectator.set_transform(carla.Transform(
                    carla.Location(x=smooth_cam_x, y=smooth_cam_y, z=smooth_cam_z),
                    carla.Rotation(pitch=-15.0, yaw=smooth_cam_yaw)
                ))

            if show_attn and attn_fig is not None:
                try:
                    attn_fig.canvas.flush_events()
                except Exception:
                    pass

    finally:
        print("\n[OPRIRE] Se opresc senzorii si conexiunea...")
        if show_attn and attn_fig is not None:
            plt.close(attn_fig)
        if camera is not None and camera.is_listening:
            camera.stop()
        cleanup_actors(world)
        pygame.quit()
        print("Exit.")


if __name__ == "__main__":
    main()