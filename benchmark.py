"""
BENCHMARK CNN vs ViT — Comparatie pe rute fixe in CARLA Town01
Utilizare:
    python benchmark.py --model cnn
    python benchmark.py --model vit

Ruleaza 6 rute (cate una per vreme), colecteaza metrici si salveaza in benchmark_results.csv.
Ambele modele trebuie rulate pe ACELEASI rute pentru comparatie corecta.
"""

import carla
import time
import os
import csv
import argparse
import numpy as np
from PIL import Image
import queue
import math
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import sys
import glob
import random
import pygame
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime

try:
    sys.path.append(glob.glob('../carla')[0])
except IndexError:
    pass

from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import RoadOption


# === CONFIGURATION ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CNN_MODEL_PATH = "model_nav_traffic.pth"
VIT_MODEL_PATH = "model_nav_traffic_vit.pth"
BENCHMARK_LOG = "benchmark_results.csv"

NUM_TRAFFIC_VEHICLES = 30
MAX_ROUTE_TIME = 300        # 5 minutes max
STUCK_TIMEOUT = 60          # 1 minute stuck = abort
STUCK_SPEED_THRESHOLD = 0.5 # km/h —under this limit = blocat
DESTINATION_THRESHOLD = 20.0 # meters — cloe enough = route complete
NPC_FREEZE_CHECK_INTERVAL = 10.0  # NPC freeze check every 10 seconds
NPC_FREEZE_TIMEOUT = 15.0         # NPC stuck > 15s = destroy
BLOCKED_BY_NPC_DIST = 8.0         # distance for detection the vehicle in front


ROUTE_DEFINITIONS = [
    {"spawn_idx": 0,   "dest_idx": 100},
    {"spawn_idx": 20,  "dest_idx": 130},
    {"spawn_idx": 40,  "dest_idx": 160},
    {"spawn_idx": 60,  "dest_idx": 180},
    {"spawn_idx": 80,  "dest_idx": 200},
    {"spawn_idx": 10,  "dest_idx": 150},
]

# weather presets for each route
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


# =====================================================================
#                        MODELS ARHITECTURE
# =====================================================================

class ConditionalNvidiaModel(nn.Module):
    """CNN NVIDIA conditionat cu GPS + semafor."""
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
        self.tl_fc = nn.Sequential(nn.Linear(3, 16), nn.ReLU())
        self.joint_fc = nn.Sequential(
            nn.Linear(1152 + 16 + 16, 256), nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, img, cmd, tl):
        img_feats = self.conv_layers(img)
        cmd_onehot = F.one_hot(cmd.long(), num_classes=4).float()
        cmd_features = self.command_fc(cmd_onehot)
        tl_onehot = F.one_hot(tl.long(), num_classes=3).float()
        tl_features = self.tl_fc(tl_onehot)
        combined = torch.cat((img_feats, cmd_features, tl_features), dim=1)
        return self.joint_fc(combined)


class ConvStem(nn.Module):
    """Conv Stem pentru Hybrid ViT."""
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
    """Hybrid ViT cu Conv Stem + GPS Token + TL Token."""
    def __init__(self, img_h=66, img_w=200,
                 embed_dim=128, num_heads=4, num_layers=4, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv_stem = ConvStem(embed_dim)
        self.num_patches_h = math.ceil(img_h / 8)
        self.num_patches_w = math.ceil(img_w / 8)
        self.num_patches = self.num_patches_h * self.num_patches_w

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.gps_embed = nn.Sequential(
            nn.Linear(4, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.tl_embed = nn.Sequential(
            nn.Linear(3, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
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
        return self.head(x[:, 0])

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


# =====================================================================
#                         UTILITARE CARLA
# =====================================================================

# === HOOKS FOR CNN HEATMAP ===
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

def carla_image_to_rgb(image):
    """Converteste imagine CARLA in numpy RGB (66x200) pentru attention overlay."""
    image.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
    pil_image = Image.fromarray(array)
    pil_cropped = pil_image.crop((0, 40, 320, 240))
    pil_resized = pil_cropped.resize((200, 66))
    return np.array(pil_resized)

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
    traffic_manager.set_global_distance_to_leading_vehicle(1.5)
    traffic_manager.global_percentage_speed_difference(-20)  # NPC-uri merg 20% mai repede
    spawned = []
    random.shuffle(spawn_points)
    for i in range(min(num_vehicles, len(spawn_points))):
        bp = random.choice(blueprint_library.filter('vehicle.*'))
        if int(bp.get_attribute('number_of_wheels')) == 4:
            npc = world.try_spawn_actor(bp, spawn_points[i])
            if npc is not None:
                npc.set_autopilot(True)
                spawned.append(npc)
    return spawned

def check_vehicle_blocking_front(vehicle, world, max_dist=BLOCKED_BY_NPC_DIST):
    """
    Verifica daca exista un vehicul NPC blocat in fata masinii noastre.
    Returneaza True daca da.
    """
    v_loc = vehicle.get_location()
    v_fwd = vehicle.get_transform().get_forward_vector()

    for npc in world.get_actors().filter('vehicle.*'):
        if npc.id == vehicle.id:
            continue
        npc_loc = npc.get_location()
        dist = v_loc.distance(npc_loc)
        if dist > max_dist:
            continue
        #check daca e in fata (dot product pozitiv)
        dx = npc_loc.x - v_loc.x
        dy = npc_loc.y - v_loc.y
        dot = dx * v_fwd.x + dy * v_fwd.y
        if dot < 0:
            continue
        # check daca NPC-ul e aproape oprit
        npc_vel = npc.get_velocity()
        npc_speed = 3.6 * math.sqrt(npc_vel.x**2 + npc_vel.y**2 + npc_vel.z**2)
        if npc_speed < 1.0:
            return True
    return False

def destroy_frozen_npcs(world, ego_vehicle, npc_speed_tracker):
    """
    Distruge NPC-urile care stau pe loc de prea mult timp.
    npc_speed_tracker: dict {npc_id: seconds_frozen}
    Returneaza numarul de NPC-uri distruse.
    """
    destroyed = 0
    current_npcs = world.get_actors().filter('vehicle.*')

    # search for NPCs tracker that no longer exists
    active_ids = {npc.id for npc in current_npcs}
    for npc_id in list(npc_speed_tracker.keys()):
        if npc_id not in active_ids:
            del npc_speed_tracker[npc_id]

    for npc in current_npcs:
        if npc.id == ego_vehicle.id:
            continue
        npc_vel = npc.get_velocity()
        npc_speed = 3.6 * math.sqrt(npc_vel.x**2 + npc_vel.y**2 + npc_vel.z**2)

        if npc_speed < 0.5:
            npc_speed_tracker[npc.id] = npc_speed_tracker.get(npc.id, 0) + NPC_FREEZE_CHECK_INTERVAL
        else:
            npc_speed_tracker[npc.id] = 0

        if npc_speed_tracker.get(npc.id, 0) > NPC_FREEZE_TIMEOUT:
            try:
                npc.destroy()
                destroyed += 1
                del npc_speed_tracker[npc.id]
            except Exception:
                pass

    return destroyed

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

def detect_traffic_light_state(vehicle, world):
    """
    Detectie semafor in doi pasi: API standard + fallback manual 30m.
    Returneaza (stare, distanta).
    """
    tl = vehicle.get_traffic_light()
    if tl is not None:
        tl_loc = tl.get_location()
        v_loc = vehicle.get_location()
        v_fwd = vehicle.get_transform().get_forward_vector()
        dx = tl_loc.x - v_loc.x
        dy = tl_loc.y - v_loc.y
        dot = dx * v_fwd.x + dy * v_fwd.y
        if dot < 0:
            return 0, 999.0
        dist = v_loc.distance(tl_loc)
        tl_state = tl.get_state()
        if tl_state == carla.TrafficLightState.Red: return 1, dist
        elif tl_state == carla.TrafficLightState.Yellow: return 2, dist
        else: return 0, dist

    # Fallback: manual scan 30 m
    v_loc = vehicle.get_location()
    v_fwd = vehicle.get_transform().get_forward_vector()
    v_wp = world.get_map().get_waypoint(v_loc)
    best_dist = 999.0
    best_state = 0
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
        if (tl_wp.lane_id * v_wp.lane_id) < 0: continue
        if dist < best_dist:
            best_dist = dist
            state = tl_actor.get_state()
            if state == carla.TrafficLightState.Red: best_state = 1
            elif state == carla.TrafficLightState.Yellow: best_state = 2
            else: best_state = 0
    return best_state, best_dist

def get_lateral_deviation(vehicle, world):
    """Distanta laterala de la vehicul la centrul benzii."""
    v_loc = vehicle.get_location()
    wp = world.get_map().get_waypoint(v_loc)
    if wp is None:
        return 0.0
    wp_loc = wp.transform.location
    wp_right = wp.transform.get_right_vector()
    dx = v_loc.x - wp_loc.x
    dy = v_loc.y - wp_loc.y
    lateral = abs(dx * wp_right.x + dy * wp_right.y)
    return lateral


# =====================================================================
#              LOGICA DE CONTROL UNIFICATA (identica pentru ambele modele)
# =====================================================================

STEERING_HISTORY_SIZE = 2
BRAKE_HISTORY_SIZE = 5
SPEED_STEER_FREE = 15.0
SPEED_STEER_MAX = 30.0
SPEED_STEER_MIN_FACTOR = 0.55
MAX_STEER_CHANGE = 0.08
THROTTLE_MULTIPLIER = 55.0


def apply_control(raw_steer, raw_throttle, raw_brake,
                  steering_history, brake_history, last_steer,
                  current_speed, current_command, current_tl_state):
    """
    Logica de control unificata — aceeasi pentru CNN si ViT.
    Doar reteaua neurala difera, post-procesarea e identica.
    Returneaza (steer, throttle, brake, safety_net_activated).
    """
    # Steering: history + speed factor + rate limiter
    steering_history.pop(0)
    steering_history.append(raw_steer)
    avg_steer = sum(steering_history) / len(steering_history)

    if current_speed < SPEED_STEER_FREE:
        speed_factor = 1.0
    elif current_speed > SPEED_STEER_MAX:
        speed_factor = SPEED_STEER_MIN_FACTOR
    else:
        t = (current_speed - SPEED_STEER_FREE) / (SPEED_STEER_MAX - SPEED_STEER_FREE)
        speed_factor = 1.0 - t * (1.0 - SPEED_STEER_MIN_FACTOR)

    desired_steer = max(-1.0, min(1.0, avg_steer * speed_factor))
    steer_diff = desired_steer - last_steer
    if abs(steer_diff) > MAX_STEER_CHANGE:
        desired_steer = last_steer + MAX_STEER_CHANGE * (1 if steer_diff > 0 else -1)
    steer = desired_steer

    # Brake: 3 zone cu history
    brake_history.pop(0)
    brake_history.append(raw_brake)
    avg_brake = sum(brake_history) / len(brake_history)

    throttle = 0.0
    brake = 0.0
    safety_net = False

    if avg_brake > 0.30:
        # if TL is green/none and car is stopped => ignore brake
        if current_tl_state != 1 and current_speed < 2.0:
            brake = 0.0
            throttle = 0.15  # pornire usoara
        else:
            brake = max(0.4, min(1.0, avg_brake * 2.0))
            throttle = 0.0
    else:
        brake = 0.0
        target_speed = raw_throttle * THROTTLE_MULTIPLIER
        if current_command == 0:   max_speed = 30.0
        elif current_command == 2: max_speed = 11.0
        elif current_command == 1: max_speed = 15.0
        else:                      max_speed = 20.0
        target_speed = min(target_speed, max_speed)
        speed_error = target_speed - current_speed
        if speed_error > 0:
            calc_throttle = speed_error * 0.15
            throttle = max(0.25, min(0.75, calc_throttle))
        elif current_speed < 3.0:
            throttle = 0.15
        else:
            throttle = 0.0

    # Safety net TL
    if current_tl_state == 1:  # ROSU
        safety_net = True
        brake = 0.8
        throttle = 0.0

    return steer, throttle, brake, safety_net


# =====================================================================
#                          CSV LOGGING
# =====================================================================

def init_benchmark_log(log_path):
    """Creeaza CSV-ul daca nu exista."""
    if os.path.exists(log_path):
        return
    header = [
        "timestamp", "model", "route_id", "weather",
        "collisions", "distance_total_m", "max_dist_no_collision_m",
        "red_light_violations", "safety_net_activations",
        "avg_lane_deviation_m", "route_completed",
        "time_seconds", "avg_speed_kmh", "abort_reason"
    ]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(header)
    print(f"[LOG] Fisier {log_path} creat.")

def save_route_result(log_path, result):
    """Adauga o linie in CSV."""
    row = [
        result["timestamp"], result["model"], result["route_id"], result["weather"],
        result["collisions"], f"{result['distance_total']:.1f}",
        f"{result['max_dist_no_collision']:.1f}",
        result["red_light_violations"], result["safety_net_activations"],
        f"{result['avg_lane_deviation']:.3f}", result["route_completed"],
        f"{result['time_seconds']:.1f}", f"{result['avg_speed']:.1f}",
        result["abort_reason"]
    ]
    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow(row)


# =====================================================================
#                      RULARE RUTA INDIVIDUALA
# =====================================================================

def run_single_route(model, model_type, world, client, spawn_points,
                     route_id, spawn_idx, dest_idx, weather_name):
    """
    Ruleaza o singura ruta si returneaza dictionarul de metrici.
    """
    print(f"\n{'='*60}")
    print(f"  RUTA {route_id + 1}/6 | Vreme: {weather_name} | Model: {model_type.upper()}")
    print(f"  Spawn: {spawn_idx} -> Dest: {dest_idx}")
    print(f"{'='*60}")

    blueprint_library = world.get_blueprint_library()

    # set weather
    world.set_weather(WEATHER_PRESETS[weather_name])

    # spawn vehicel
    vehicle_bp = blueprint_library.filter("model3")[0]
    num_sp = len(spawn_points)
    actual_spawn_idx = spawn_idx % num_sp
    actual_dest_idx = dest_idx % num_sp

    vehicle = None
    attempts = 0
    while vehicle is None and attempts < 10:
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[actual_spawn_idx])
        if vehicle is None:
            actual_spawn_idx = (actual_spawn_idx + 1) % num_sp
            attempts += 1
    if vehicle is None:
        print(f"  [EROARE] Nu s-a putut spawna vehiculul!")
        return None
    time.sleep(1.0)

    # Agent pentru ruta
    agent = BasicAgent(vehicle, target_speed=20)
    agent.ignore_traffic_lights(active=True)
    agent.ignore_stop_signs(active=True)
    dest_location = spawn_points[actual_dest_idx].location
    agent.set_destination(dest_location)

    # Camera
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", "320")
    camera_bp.set_attribute("image_size_y", "240")
    camera_bp.set_attribute("fov", "90")
    camera_bp.set_attribute("sensor_tick", "0.1")
    cam_transform = carla.Transform(carla.Location(x=1.5, z=1.4), carla.Rotation(pitch=-15.0))
    camera = world.spawn_actor(camera_bp, cam_transform, attach_to=vehicle)
    image_queue = queue.Queue()
    camera.listen(image_queue.put)

    # collision sensor
    collision_bp = blueprint_library.find("sensor.other.collision")
    collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
    collision_count = 0
    active_collisions = {}  # {actor_id: last_event_time} — obiecte atinse activ
    COLLISION_SEPARATION_TIME = 2.0  # dupa 2s fara contact = s-a separat

    def on_collision(event):
        nonlocal collision_count, current_dist_no_collision, max_dist_no_collision
        actor_id = event.other_actor.id
        now = time.time()
        if actor_id not in active_collisions:
            # collision with new object => new collision
            collision_count += 1
            if current_dist_no_collision > max_dist_no_collision:
                max_dist_no_collision = current_dist_no_collision
            current_dist_no_collision = 0.0
            print(f"  [!] Coliziune #{collision_count} cu {event.other_actor.type_id}")
        # actualizare timestamp-ul pentru acest actor (inca in contact)
        active_collisions[actor_id] = now

    collision_sensor.listen(on_collision)

    # --- State pentru control ---
    steering_history = [0.0] * STEERING_HISTORY_SIZE
    brake_history = [0.0] * BRAKE_HISTORY_SIZE
    last_steer = 0.0
    last_throttle = 0.0
    last_brake = 0.0

    # --- Metrici ---
    total_distance = 0.0
    current_dist_no_collision = 0.0
    max_dist_no_collision = 0.0

    safety_net_activations = 0       # frames or safety net
    red_light_encounters = 0         # total of red lights encounters
    red_light_violations = 0         # red light violations
    prev_tl_state = 0
    prev_tl_dot = -1.0               # dot product la ultimul semafor rosu

    lane_deviations = []
    speed_samples = []
    current_tl_state = 0
    lat_dev = 0.0

    prev_location = vehicle.get_location()
    stuck_timer = 0.0
    npc_block_timer = 0.0  # timp blocat de NPC (nu se numara ca stuck propriu)
    route_start_time = time.time()
    last_npc_check_time = time.time()
    npc_speed_tracker = {}  # {npc_id: seconds_frozen}
    abort_reason = "none"
    route_completed = False
    frame_counter = 0

    print(f"  [START] Distanta pana la destinatie: {prev_location.distance(dest_location):.0f}m")

    # --- SPECTATOR CAMERA ---
    spectator = world.get_spectator()
    CAMERA_SMOOTH = 0.1
    smooth_cam_x = None
    smooth_cam_y = None
    smooth_cam_z = None
    smooth_cam_yaw = None
    follow_mode = False  

    # --- ATTENTION MAP / HEATMAP STATE ---
    show_attn = False
    m_pressed_last_frame = False
    attn_fig = None
    attn_ax = None
    attn_im = None
    last_raw_image = None
    active_layer = 4 if model_type == "vit" else 5  
    max_layers = 4 if model_type == "vit" else 5
    smooth_attn = None

    # --- PYGAME DISPLAY ---
    pygame.init()
    pg_display = pygame.display.set_mode((580, 230))
    pygame.display.set_caption(f"Benchmark {model_type.upper()} | Ruta {route_id + 1}")
    font = pygame.font.SysFont("consolas", 15)
    clock = pygame.time.Clock()

    try:
        while True:
            clock.tick(60)
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    abort_reason = "user_skip"
                    break
            keys = pygame.key.get_pressed()

            # ESC — skip route
            if keys[pygame.K_ESCAPE]:
                abort_reason = "user_skip"
                print(f"  [SKIP] Ruta sarita de utilizator")
                break

            # V — toggle camera follow
            if keys[pygame.K_v]:
                follow_mode = not follow_mode
                time.sleep(0.3)

            # M — toggle heatmap/attention map
            if keys[pygame.K_m] and not m_pressed_last_frame:
                show_attn = not show_attn
                if show_attn:
                    plt.ion()
                    attn_fig, attn_ax = plt.subplots(1, 1, figsize=(8, 3))
                    vis_type = "Attention Map" if model_type == "vit" else "Conv Heatmap"
                    layer_label = f"Layer {active_layer}" if model_type == "vit" else f"Conv{active_layer}"
                    attn_fig.canvas.manager.set_window_title(f"Benchmark — {vis_type} ({model_type.upper()})")
                    attn_ax.set_title(f"{vis_type} — {layer_label}", fontsize=12)
                    attn_ax.axis('off')
                    dummy = np.zeros((66, 200, 3), dtype=np.uint8)
                    attn_im = attn_ax.imshow(dummy)
                    plt.tight_layout()
                    plt.show(block=False)
                    smooth_attn = None
                    print(f"  [{vis_type.upper()}] ON — {layer_label}. [1-{max_layers}] Schimba strat")
                else:
                    if attn_fig is not None:
                        plt.close(attn_fig)
                        attn_fig = None
                        attn_ax = None
                        attn_im = None
                    smooth_attn = None
                    print("  [HEATMAP] OFF")
            m_pressed_last_frame = keys[pygame.K_m]

            # 1-5 — change layer (CNN: 1-5, ViT: 1-4)
            for key, layer_num in [(pygame.K_1, 1), (pygame.K_2, 2), (pygame.K_3, 3), (pygame.K_4, 4), (pygame.K_5, 5)]:
                if keys[key] and layer_num <= max_layers and active_layer != layer_num:
                    active_layer = layer_num
                    smooth_attn = None
                    if show_attn and attn_ax is not None:
                        if model_type == "vit":
                            attn_ax.set_title(f"CLS Attention — Layer {active_layer}", fontsize=12)
                        else:
                            attn_ax.set_title(f"Conv Heatmap — Conv{active_layer}", fontsize=12)
                    print(f"  [LAYER] {active_layer}/{max_layers}")
                    time.sleep(0.2)
                    break

            # max time check
            elapsed = time.time() - route_start_time
            if elapsed > MAX_ROUTE_TIME:
                abort_reason = "timeout_max"
                print(f"  [ABORT] Timp maxim depasit ({MAX_ROUTE_TIME}s)")
                break

            # check if the destination was achieved
            current_loc = vehicle.get_location()
            dist_to_dest = current_loc.distance(dest_location)
            if dist_to_dest < DESTINATION_THRESHOLD:
                route_completed = True
                print(f"  [SUCCES] Destinatie atinsa! ({dist_to_dest:.1f}m)")
                break

            # current speed
            vel = vehicle.get_velocity()
            current_speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

            # Agent — route and GPS 
            if agent.done():
                agent.set_destination(dest_location)
            auto_control = agent.run_step()
            current_road_option = agent.get_local_planner().target_road_option
            current_command = map_command(current_road_option)

            # Detectie semafor (INAINTE de stuck check)
            current_tl_state, tl_dist = detect_traffic_light_state(vehicle, world)


            # === INFERENTA MODEL (prioritate maxima) ===
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

                    steer, throttle, brake, safety_activated = apply_control(
                        raw_steer, raw_throttle, raw_brake,
                        steering_history, brake_history, last_steer,
                        current_speed, current_command, current_tl_state
                    )
                    last_steer = steer
                    last_throttle = throttle
                    last_brake = brake

                    if safety_activated:
                        safety_net_activations += 1

                    control = carla.VehicleControl()
                    control.steer = steer
                    control.throttle = throttle
                    control.brake = brake
                    vehicle.apply_control(control)

                    # --- HEATMAP / ATTENTION MAP UPDATE ---
                    if show_attn and attn_fig is not None and last_raw_image is not None:
                        try:
                            rgb_frame = carla_image_to_rgb(last_raw_image)

                            if model_type == "vit":
                                # ViT: attention maps
                                raw_attn = model.get_attention_maps(layer_idx=active_layer - 1)
                            else:
                                # CNN: conv activation maps
                                layer_name = CONV_LAYER_NAMES[active_layer]
                                if layer_name in activation:
                                    act = activation[layer_name].squeeze().cpu().numpy()
                                    raw_attn = np.mean(act, axis=0)
                                    raw_attn = raw_attn - raw_attn.min()
                                    if raw_attn.max() > 0:
                                        raw_attn = raw_attn / raw_attn.max()
                                else:
                                    raw_attn = None

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
                                overlay = 0.5 * rgb_normalized + 0.5 * heatmap_colored
                                overlay = np.clip(overlay, 0, 1)
                                attn_im.set_data(overlay)
                                attn_fig.canvas.draw_idle()
                                attn_fig.canvas.flush_events()
                        except Exception:
                            pass

            except queue.Empty:
                pass

            # === OPERATII PERIODICE (la fiecare al 10lea frame) ===
            frame_counter += 1
            if frame_counter % 10 == 0:
                # Calcul distanta parcursa
                dx = current_loc.x - prev_location.x
                dy = current_loc.y - prev_location.y
                dz = current_loc.z - prev_location.z
                frame_dist = math.sqrt(dx**2 + dy**2 + dz**2)
                total_distance += frame_dist
                current_dist_no_collision += frame_dist
                prev_location = current_loc

                # Deviere laterala
                lat_dev = get_lateral_deviation(vehicle, world)
                lane_deviations.append(lat_dev)
                speed_samples.append(current_speed)

                # Curatare coliziuni expirate
                now_clean = time.time()
                expired = [aid for aid, t in active_collisions.items() if now_clean - t > COLLISION_SEPARATION_TIME]
                for aid in expired:
                    del active_collisions[aid]

                # Tracking treceri pe rosu
                if current_tl_state == 1 and prev_tl_state != 1:
                    red_light_encounters += 1
                if current_tl_state == 1 and tl_dist < 30.0:
                    tl = vehicle.get_traffic_light()
                    if tl is not None:
                        tl_loc = tl.get_location()
                        v_loc = vehicle.get_location()
                        v_fwd = vehicle.get_transform().get_forward_vector()
                        current_dot = (tl_loc.x - v_loc.x) * v_fwd.x + (tl_loc.y - v_loc.y) * v_fwd.y
                        if prev_tl_dot > 0 and current_dot < 0 and current_speed > 2.0:
                            red_light_violations += 1
                            print(f"  [!] Trecere pe ROSU detectata!")
                        prev_tl_dot = current_dot
                    else:
                        prev_tl_dot = -1.0
                else:
                    prev_tl_dot = -1.0
                prev_tl_state = current_tl_state

            # === STUCK DETECTION (la fiecare al 30lea frame) ===
            if frame_counter % 30 == 0:
                now = time.time()
                if now - last_npc_check_time > NPC_FREEZE_CHECK_INTERVAL:
                    destroyed = destroy_frozen_npcs(world, vehicle, npc_speed_tracker)
                    if destroyed > 0:
                        print(f"  [NPC] {destroyed} vehicul(e) inghetat(e) distruse")
                    last_npc_check_time = now

                if current_speed < STUCK_SPEED_THRESHOLD:
                    if current_tl_state == 1:
                        stuck_timer = 0.0
                        npc_block_timer = 0.0
                    elif check_vehicle_blocking_front(vehicle, world):
                        npc_block_timer += 0.5
                        stuck_timer = 0.0
                    else:
                        stuck_timer += 0.5
                        npc_block_timer = 0.0
                else:
                    stuck_timer = 0.0
                    npc_block_timer = 0.0

                if stuck_timer > STUCK_TIMEOUT:
                    abort_reason = "stuck_model"
                    print(f"  [ABORT] Vehicul blocat > {STUCK_TIMEOUT}s (vina modelului)")
                    break
                if npc_block_timer > STUCK_TIMEOUT:
                    abort_reason = "blocked_by_npc"
                    print(f"  [ABORT] Vehicul blocat > {STUCK_TIMEOUT}s de un NPC inghetat")
                    break


            # === DISPLAY PYGAME ===
            pg_display.fill((30, 30, 30))
            cmd_str = ["LANE", "LEFT", "RIGHT", "STRAIGHT"][current_command]
            tl_labels = ["VERDE", "ROSU", "GALBEN"]
            tl_colors = [(0, 255, 0), (255, 0, 0), (255, 255, 0)]

            if show_attn:
                vis_name = "ATTN" if model_type == "vit" else "HEATMAP"
                layer_label = f"Layer {active_layer}/{max_layers}"
                attn_status = f"{vis_name}: ON {layer_label} | [1-{max_layers}] Schimba"
                attn_color = (0, 255, 0)
            else:
                attn_status = f"[M] Heatmap OFF"
                attn_color = (100, 100, 100)

            lines = [
                (f"Model: {model_type.upper()} | Ruta {route_id+1}/6 | {weather_name}", (255,255,255)),
                (f"GPS: {cmd_str} | Speed: {current_speed:.0f} km/h | Dest: {dist_to_dest:.0f}m", (100,200,255)),
                (f"S: {last_steer:.2f} | T: {last_throttle:.2f} | B: {last_brake:.2f} | Coliziuni: {collision_count} | TL rosu: {red_light_violations}", (255,255,255)),
                (f"SEMAFOR: {tl_labels[current_tl_state]}", tl_colors[current_tl_state]),
                (f"Timp: {elapsed:.0f}s | Stuck: {stuck_timer:.0f}s | NPC block: {npc_block_timer:.0f}s", (180,180,180)),
                (f"Distanta: {total_distance:.0f}m | Dev banda: {lat_dev:.2f}m", (180,180,180)),
                (attn_status, attn_color),
                (f"[ESC] Skip | [V] Camera | [M] Attention", (100,100,100)),
            ]
            for i, (text, color) in enumerate(lines):
                pg_display.blit(font.render(text, True, color), (10, 8 + i * 25))
            pygame.display.flip()

            # === SPECTATOR CAMERA FOLLOW (la fiecare al 3-lea frame) ===
            if follow_mode and vehicle.is_alive and frame_counter % 3 == 0:
                t = vehicle.get_transform()
                target_loc = t.location - 6 * t.get_forward_vector() + carla.Location(z=3.0)
                target_yaw = t.rotation.yaw

                if smooth_cam_x is None:
                    smooth_cam_x = target_loc.x
                    smooth_cam_y = target_loc.y
                    smooth_cam_z = target_loc.z
                    smooth_cam_yaw = target_yaw
                else:
                    smooth_cam_x += (target_loc.x - smooth_cam_x) * CAMERA_SMOOTH
                    smooth_cam_y += (target_loc.y - smooth_cam_y) * CAMERA_SMOOTH
                    smooth_cam_z += (target_loc.z - smooth_cam_z) * CAMERA_SMOOTH
                    yaw_diff = target_yaw - smooth_cam_yaw
                    if yaw_diff > 180: yaw_diff -= 360
                    elif yaw_diff < -180: yaw_diff += 360
                    smooth_cam_yaw += yaw_diff * CAMERA_SMOOTH

                spectator.set_transform(carla.Transform(
                    carla.Location(x=smooth_cam_x, y=smooth_cam_y, z=smooth_cam_z),
                    carla.Rotation(pitch=-15.0, yaw=smooth_cam_yaw)
                ))

            # Flush attention events
            if show_attn and attn_fig is not None:
                try:
                    attn_fig.canvas.flush_events()
                except Exception:
                    pass

    finally:
        # Update max dist fara coliziune
        if current_dist_no_collision > max_dist_no_collision:
            max_dist_no_collision = current_dist_no_collision

        # Cleanup attention map
        if attn_fig is not None:
            try:
                plt.close(attn_fig)
            except Exception:
                pass

        # Cleanup senzori
        if collision_sensor.is_listening:
            collision_sensor.stop()
        collision_sensor.destroy()
        if camera.is_listening:
            camera.stop()
        camera.destroy()
        vehicle.destroy()
        pygame.quit()

    elapsed_total = time.time() - route_start_time
    avg_speed = sum(speed_samples) / len(speed_samples) if speed_samples else 0.0
    avg_deviation = sum(lane_deviations) / len(lane_deviations) if lane_deviations else 0.0

    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": model_type.upper(),
        "route_id": route_id + 1,
        "weather": weather_name,
        "collisions": collision_count,
        "distance_total": total_distance,
        "max_dist_no_collision": max_dist_no_collision,
        "red_light_violations": red_light_violations,
        "safety_net_activations": safety_net_activations,
        "avg_lane_deviation": avg_deviation,
        "route_completed": 1 if route_completed else 0,
        "time_seconds": elapsed_total,
        "avg_speed": avg_speed,
        "abort_reason": abort_reason,
    }

    print(f"\n  --- REZULTAT RUTA {route_id + 1} ---")
    print(f"  Coliziuni: {collision_count}")
    print(f"  Distanta totala: {total_distance:.0f}m")
    print(f"  Max dist fara coliziune: {max_dist_no_collision:.0f}m")
    print(f"  Treceri pe rosu: {red_light_violations}")
    print(f"  Safety net activari: {safety_net_activations}")
    print(f"  Deviatie medie banda: {avg_deviation:.3f}m")
    print(f"  Ruta completa: {'DA' if route_completed else 'NU'}")
    print(f"  Timp: {elapsed_total:.0f}s | Viteza medie: {avg_speed:.1f} km/h")
    if abort_reason != "none":
        print(f"  Motiv abort: {abort_reason}")

    return result


# =====================================================================
#                         BENCHMARK PRINCIPAL
# =====================================================================

def run_benchmark(model_type):
    print("\n" + "=" * 60)
    print(f"  BENCHMARK — Model: {model_type.upper()}")
    print(f"  Device: {DEVICE}")
    print(f"  Rute: {len(ROUTE_DEFINITIONS)} | Trafic: {NUM_TRAFFIC_VEHICLES} NPC")
    print("=" * 60)

    # Incarcare model
    if model_type == "cnn":
        model_path = CNN_MODEL_PATH
        model = ConditionalNvidiaModel().to(DEVICE)
    else:
        model_path = VIT_MODEL_PATH
        model = ConditionalViTModel().to(DEVICE)

    if not os.path.exists(model_path):
        print(f"\n[EROARE] Nu exista {model_path}!")
        return
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        model.eval()
        print(f"\n[OK] Model {model_path} incarcat pe {DEVICE}")

        # Inregistrare hooks pentru CNN heatmap
        if model_type == "cnn":
            for layer_num, layer_idx in CONV_LAYER_INDICES.items():
                layer_name = CONV_LAYER_NAMES[layer_num]
                model.conv_layers[layer_idx].register_forward_hook(get_activation(layer_name))
            print(f"[OK] Hooks inregistrate pe cele 5 straturi conv")
    except Exception as e:
        print(f"\n[EROARE] Eroare la incarcare: {e}")
        return

    # Conectare CARLA
    print("\nSe conecteaza la simulator...")
    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    world = client.get_world()

    current_map = world.get_map().name
    if not current_map.endswith('Town01'):
        world = client.load_world('Town01')
        time.sleep(2.0)

    init_benchmark_log(BENCHMARK_LOG)
    spawn_points = world.get_map().get_spawn_points()
    print(f"[INFO] Town01: {len(spawn_points)} spawn points disponibile")

    results = []

    for route_id, route_def in enumerate(ROUTE_DEFINITIONS):
        weather_name = WEATHER_NAMES[route_id % len(WEATHER_NAMES)]

        # Cleanup inainte de fiecare ruta
        cleanup_actors(world)
        time.sleep(1.0)

        # Spawn trafic proaspat pentru fiecare ruta
        random.seed(42 + route_id)
        spawn_traffic(world, client, NUM_TRAFFIC_VEHICLES)
        time.sleep(2.0)

        result = run_single_route(
            model=model,
            model_type=model_type,
            world=world,
            client=client,
            spawn_points=spawn_points,
            route_id=route_id,
            spawn_idx=route_def["spawn_idx"],
            dest_idx=route_def["dest_idx"],
            weather_name=weather_name,
        )

        if result is not None:
            results.append(result)
            save_route_result(BENCHMARK_LOG, result)
            print(f"  [SALVAT] Ruta {route_id + 1} salvata in {BENCHMARK_LOG}")

    # Cleanup final
    cleanup_actors(world)

    # === SUMAR FINAL ===
    print("\n" + "=" * 60)
    print(f"  SUMAR BENCHMARK — {model_type.upper()}")
    print("=" * 60)

    if results:
        total_collisions = sum(r["collisions"] for r in results)
        avg_deviation = sum(r["avg_lane_deviation"] for r in results) / len(results)
        routes_completed = sum(r["route_completed"] for r in results)
        total_violations = sum(r["red_light_violations"] for r in results)
        total_safety = sum(r["safety_net_activations"] for r in results)
        avg_speed = sum(r["avg_speed"] for r in results) / len(results)
        total_distance = sum(r["distance_total"] for r in results)

        print(f"  Rute completate: {routes_completed}/{len(results)}")
        print(f"  Coliziuni totale: {total_collisions}")
        print(f"  Distanta totala: {total_distance:.0f}m")
        print(f"  Treceri pe rosu: {total_violations}")
        print(f"  Safety net activari (frame-uri): {total_safety}")
        print(f"  Deviatie medie banda: {avg_deviation:.3f}m")
        print(f"  Viteza medie: {avg_speed:.1f} km/h")

    print(f"\n  Rezultate salvate in: {BENCHMARK_LOG}")
    print(f"  Ruleaza cu --model {'vit' if model_type == 'cnn' else 'cnn'} pentru celalalt model.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark CNN vs ViT in CARLA Town01")
    parser.add_argument("--model", choices=["cnn", "vit"], required=True,
                        help="Modelul de testat: 'cnn' sau 'vit'")
    args = parser.parse_args()
    run_benchmark(args.model)