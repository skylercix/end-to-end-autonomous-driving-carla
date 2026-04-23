## CARLA End-to-End Driving – Traffic & Navigation Aware

This folder contains an end-to-end pipeline for training and running a **traffic-aware, navigation‑conditioned driving model** in the CARLA simulator.  
The model predicts **steering, throttle and brake** from a front RGB camera and a high‑level navigation command, and is trained on data with **dense NPC traffic**:

- **data collection** in traffic with hybrid manual/autopilot control and navigation commands
- **dataset processing** with smoothing, balancing, crop/resize, flip augmentation and detailed statistics
- **multi‑output model training** (conditional NVIDIA‑style CNN → steer, throttle, brake)
- **feature‑map visualization** of the first conv layer
- **autonomous driving** with traffic using the trained model

All scripts assume they are run from this `examples` folder so that relative paths to datasets and the model file work correctly.

---

## 1. Environment & Virtual Environment

These examples assume you already have (and **always use**):

- **Conda** installed
- A Conda environment named **`carla-gpu`**
- **CARLA simulator** running and listening on `localhost:2000` (all scripts are hard‑coded to this host/port)

### 1.1. Activate the Conda environment

In a terminal, from anywhere:

```bash
conda activate carla-gpu
```

Then change into this folder:

```bash
cd path/to/PythonAPI/examples
```

All the commands below should be run from this `examples` directory.

### 1.2. (Optional) Minimal dependency checklist

The environment should contain at least:

- `carla` Python package (matching your CARLA version)
- `torch`, `torchvision`
- `numpy`, `pillow`
- `pygame`
- `matplotlib`

### 1.3. Exact versions (Python, CARLA, libs)

This project is configured and tested with the following versions:

- **Python**: `3.8.20`
- **CARLA**: `0.9.12` (server and Python API)
- **`future`**: version not pinned (any compatible version)
- **`numpy`**:
  - `numpy==1.18.4` for Python 3 (`python_version >= '3.0'`)
- **`pygame`**, **`matplotlib`**, **`Pillow`**, **`open3d`**: versions not pinned (use recent stable releases)

All experiments in this guide assume the environment above.

---

## 2. Script overview

### 2.1. `collect_autopilot.py` – Data collection with navigation commands and traffic

**Purpose**: Collect driving data in **Town01** with both **camera images**, **high‑level navigation commands** and **dense NPC traffic**, while allowing you to switch between CARLA autopilot and manual driving.

- Connects to CARLA on `localhost:2000`.
- Ensures map `Town01` is loaded.
- Spawns **NPC traffic** (30–40 vehicles) using CARLA’s `TrafficManager` so you get realistic urban interactions (following distance, braking for other cars, etc.).
- Spawns your ego `model3` vehicle and an RGB camera on the hood.
- Uses CARLA’s `BasicAgent` to compute a **navigation route** and corresponding `RoadOption` (LEFT / RIGHT / STRAIGHT / LANE).
- Saves data into `dataset_traffic/episode_XXX/` folders:
  - images: `frame_id.png`
  - labels: `controls_nav.csv` containing `filename, steer, throttle, brake, command`, where:
    - `command = 0` → LANE (keep lane / no specific turn)
    - `command = 1` → LEFT
    - `command = 2` → RIGHT
    - `command = 3` → STRAIGHT
- Displays a Pygame window with:
  - current driver mode (**AUTO** vs **MANUAL**)
  - recording status
  - GPS command, steering, throttle and brake
  - a 2D **radar‑style visualization** of the upcoming waypoints (route).

**Driver modes & controls**:

- **Mode toggle**:
  - `M`: toggle between **autopilot** and **manual (WASD)** driving.
- **Manual driving (when manual mode is active)**:
  - `W` / `S`: throttle / brake
  - `A` / `D`: steer left / right (smoothed over time)
- **Recording episodes**:
  - `SPACE` (hold): start/continue recording **one episode**
  - `SPACE` (release): stop recording and write `controls_nav.csv` for that episode
- **Exit**:
  - `ESC`: exit script

Images (with corresponding control and command values) are saved **while you hold SPACE**, including when slowing down or stopping in traffic.

**Usage**:

```bash
conda activate carla-gpu
cd path/to/PythonAPI/examples
python collect_autopilot.py
```

You should see folders like `dataset_traffic/episode_000`, `episode_001`, etc., each with images and a `controls_nav.csv` file including steering, throttle, brake and navigation command.

---

### 2.2. `process_data.py` – Smooth, balance, crop, resize, flip and analyze the traffic dataset

**Purpose**: Take the raw traffic‑aware dataset in `dataset_traffic/` and create a **clean, smoothed, balanced and structurally preprocessed dataset** in `dataset_traffic_processed/` for training.

- Reads each `controls_nav.csv` from `dataset_traffic/episode_XXX/`:
  - expects columns: `filename, steer, throttle, brake, command`.
- Applies a **moving‑average smoothing** to the steering signal (window size `SMOOTHING_WINDOW`) to turn discrete keyboard/autopilot inputs into a smoother steering curve.
- Reduces the number of almost‑straight frames **without discarding braking events**:
  - if `abs(new_steer) < 0.05` **and** `brake < 0.1`, each frame is kept with probability `KEEP_PROBABILITY` (default 0.1) to balance turns vs straight driving.
- Performs **image‑structure preprocessing**:
  - crops out the sky/top of the original `320×240` image (`crop((0, 80, 320, 240))`)
  - resizes the cropped region to `200×66` (width × height)
  - saves only the processed images.
- Performs **data augmentation via horizontal flips**:
  - with probability `FLIP_PROBABILITY` (default 0.5), creates a flipped copy of an image:
    - `steer` is negated
    - `command` is swapped LEFT ↔ RIGHT (1 ↔ 2); STRAIGHT and LANE remain unchanged
    - throttle and brake are kept identical.
- Writes new `controls_nav.csv` files in `dataset_traffic_processed/episode_XXX/` with the updated steering, throttle, brake and commands.
- Builds a **dashboard of histograms** to understand the final dataset:
  - original vs processed steering distribution
  - throttle distribution
  - brake distribution
  - counts per navigation command (LANE / LEFT / RIGHT / STRAIGHT).

Key configuration at the top of the file:

- `INPUT_DIR = "dataset_traffic"`
- `OUTPUT_DIR = "dataset_traffic_processed"`
- `SMOOTHING_WINDOW = 5`
- `KEEP_PROBABILITY = 0.1`
- `FLIP_PROBABILITY = 0.5`
- `FINAL_W, FINAL_H = 200, 66`

**Usage**:

```bash
conda activate carla-gpu
cd path/to/PythonAPI/examples
python process_data.py
```

This will remove any existing `dataset_traffic_processed/` and recreate it from `dataset_traffic/`.  
**Re‑run this script every time you add new episodes to `dataset_traffic/`.**

---

### 2.3. `train.py` – Train the traffic‑aware, navigation‑conditioned driving model (3 outputs)

**Purpose**: Train a **conditional NVIDIA‑style CNN** (`ConditionalNvidiaModel`) that predicts **steering, throttle and brake** from preprocessed images **and** the high‑level navigation command.

- Uses dataset from `dataset_traffic_processed/episode_XXX/` (generated by `process_data.py`):
  - reads each `controls_nav.csv` with smoothed steering, throttle, brake and navigation command.
  - matches image paths with `[steer, throttle, brake]` + `command`.
- `CarlaNavDataset`:
  - iterates over all `episode_XXX` folders in `DATASET_DIR`.
  - for each row in `controls_nav.csv`, loads the **already cropped + resized** image (`3×66×200`).
  - converts the image to **YUV/YCbCr** and then to tensor.
  - returns:
    - image tensor (`3×66×200`)
    - command as a scalar float tensor
    - target vector `[steer, throttle, brake]` as a 3‑D float tensor.
- `ConditionalNvidiaModel`:
  - image branch: classic NVIDIA CNN stack:
    - conv layers: `(3→24)`, `(24→36)`, `(36→48)`, `(48→64)`, `(64→64)` with strides and ReLUs, then flatten.
  - command branch:
    - fully‑connected `1 → 16` with ReLU.
  - joint head:
    - concatenates image features and command features
    - fully‑connected layers `1152+16 → 256 → 128 → 64 → 3`  
      output is a 3‑vector: `[steer, throttle, brake]`.
- Training parameters (top of file, current defaults):
  - `DATASET_DIR = "dataset_traffic_processed"`
  - `MODEL_SAVE_PATH = "model_nav_traffic.pth"`
  - `BATCH_SIZE = 64`
  - `NUM_EPOCHS = 40`
  - `LEARNING_RATE = 3e-4`
  - validation split: `VAL_SPLIT = 0.15`
  - device selection:
    - `DEVICE = "cuda"` if available, otherwise `"cpu"`.
- Uses **MSE loss** on the 3‑component output and **Adam** optimizer.
- Uses a **ReduceLROnPlateau** scheduler on validation loss.
- Tracks training/validation loss and:
  - saves the **best model checkpoint** as `model_nav_traffic.pth`
  - saves a plot of the loss curves as `training_history.png` and also shows it.

Example training history (MSE loss vs. epochs):



**Usage**:

1. Make sure you already have data in `dataset_traffic/` (from `collect_autopilot.py`) and that you have run `process_data.py` so that `dataset_traffic_processed/` exists.
2. Run:

```bash
conda activate carla-gpu
cd path/to/PythonAPI/examples
python train.py
```

After training, you should see `model_nav_traffic.pth` and `training_history.png` created.

---

### 2.4. `feature_map.py` – Visualize feature maps of the conditional traffic model

**Purpose**: Visualize the feature maps from the **first convolutional layer** of the traffic‑aware model to understand what it is learning.

- Loads the same architecture (`ConditionalNvidiaModel`) and weights from `model_nav_traffic.pth`.
- Samples a random preprocessed image from `dataset_traffic_processed/`.
- Applies the same **YUV + tensor** preprocessing as used in training (images are already cropped/resized by `process_data.py`).
- Creates a **dummy navigation command** (e.g. STRAIGHT = `3.0`) for visualization.
- Registers a forward hook on the **first convolutional layer** (`conv_layers[0]`).
- Runs a forward pass and extracts activations of that layer.
- Displays with `matplotlib`:
  - the input image (in YUV space)
  - each feature map of the first conv layer in a grid.

**Usage**:

```bash
conda activate carla-gpu
cd path/to/PythonAPI/examples
python feature_map.py
```

This will open a matplotlib window showing the input and the feature maps of the first convolutional layer.

---

### 2.5. `drive_model.py` – Autonomous driving with traffic using the 3‑output model

**Purpose**: Use the trained `ConditionalNvidiaModel` (`model_nav_traffic.pth`) to **drive the car autonomously in traffic**, conditioned on the navigation command from `BasicAgent`.

- Loads `model_nav_traffic.pth` on GPU if available (`DEVICE = "cuda"` when possible).
- Ensures map `Town01` is loaded.
- Spawns **NPC traffic** (via `spawn_traffic`) before spawning the ego vehicle.
- Spawns a `model3` ego vehicle and an RGB camera.
- Uses `BasicAgent` to compute a route and obtain the current `RoadOption` at every step.
- Maps `RoadOption` to a numeric `command`:
  - `0` → LANE
  - `1` → LEFT
  - `2` → RIGHT
  - `3` → STRAIGHT
- Preprocesses camera images with the **same pipeline used in training**:
  - BGRA → RGB
  - crop top part (road‑focused)
  - convert to YUV/YCbCr
  - resize to `66×200`
  - convert to tensor and move to `DEVICE`.
- Maintains a **steering history buffer** (`STEERING_HISTORY_SIZE`) to smooth predicted steering over time.
- For each new frame:
  - converts the latest image to tensor
  - feeds `(image, command)` to the model
  - obtains `steer`, `throttle`, `brake` predictions
  - smooths steering across recent predictions
  - applies a **simple speed controller**:
    - converts `raw_throttle` into a target speed
    - computes a proportional control on the speed error
    - uses the model’s `brake` output to strongly brake when needed and cut throttle.
- The Pygame window shows:
  - current model file and device
  - GPS command and the current applied `steer / throttle / brake`
  - controls hint (`[V] Camera | [R] Respawn | [ESC] Exit`)
  - a radar‑like 2D view of waypoints around the vehicle (same style as data collection).
- The spectator camera can follow the car for a third‑person view.

**Controls**:

- `V`: toggle spectator camera mode (follow / free).
- `R`: respawn the ego vehicle at a new random spawn point and regenerate the route.
- `ESC`: stop the script and clean up.

**Usage**:

1. Make sure `model_nav_traffic.pth` exists (trained via `train.py`).
2. Ensure CARLA (Town01) is running.
3. Run:

```bash
conda activate carla-gpu
cd path/to/PythonAPI/examples
python drive_model.py
```

---

## 3. Recommended workflow (step‑by‑step)

1. **Start CARLA simulator**
   - Launch CARLA and ensure the server is running on `localhost:2000` (map `Town01` will be loaded or reloaded by the scripts if needed).

2. **Activate environment and move into examples**

   ```bash
   conda activate carla-gpu
   cd path/to/PythonAPI/examples
   ```

3. **Collect traffic‑aware, navigation‑conditioned training data (main dataset: `dataset_traffic`)**
   - Run `python collect_autopilot.py`.
   - Wait for NPC traffic to spawn.
   - Use `M` to toggle between autopilot and manual `W/A/S/D` control.
   - Hold `SPACE` to record an episode; release `SPACE` to stop and save it as `episode_XXX` with `controls_nav.csv`.
   - Repeat to gather multiple episodes in `dataset_traffic/` (you can mix manual and autopilot control, and interact with traffic).

4. **Process the data into `dataset_traffic_processed`**

   ```bash
   conda activate carla-gpu
   cd path/to/PythonAPI/examples
   python process_data.py
   ```

   - This will create/update `dataset_traffic_processed/` from your latest episodes (smooth, balanced, cropped, resized, flipped, with histograms for steer/throttle/brake/command).

5. **Train the 3‑output navigation‑aware model (on `dataset_traffic_processed`)**

   ```bash
   conda activate carla-gpu
   cd path/to/PythonAPI/examples
   python train.py
   ```

   - Wait until training finishes and `model_nav_traffic.pth` and `training_history.png` are saved.

6. **(Optional) Inspect learned features on traffic data**

   ```bash
   conda activate carla-gpu
   cd path/to/PythonAPI/examples
   python feature_map.py
   ```

   - A window will show the input image and the first‑layer feature maps of the conditional model trained in traffic.

7. **Run autonomous driving with the traffic‑aware model**

   ```bash
   conda activate carla-gpu
   cd path/to/PythonAPI/examples
   python drive_model.py
   ```

   - Use `V` to toggle camera follow mode and `R` to respawn the ego vehicle if needed.

---

## 4. Things you may want to adjust

- **Dataset folders**: the main pipeline is now  
  `dataset_traffic` → `dataset_traffic_processed` → `model_nav_traffic.pth`.  
  If you introduce alternative raw datasets (e.g. different traffic densities or towns), you can:
  - change `INPUT_DIR` in `process_data.py` to the alternative dataset, and/or
  - change `DATASET_DIR` in `train.py` / `DATASET_DIR` in `feature_map.py`
  so they use the corresponding processed dataset.
- **Training hyperparameters**: adjust `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`, `VAL_SPLIT`, `NUM_WORKERS` and `PREFETCH_FACTOR` in `train.py` to fit your GPU/CPU and dataset size.
- **Camera resolution / FOV / crop**: the CARLA camera settings are in `collect_autopilot.py` and `drive_model.py`. The **structural preprocessing** (crop + resize) is centralized in `process_data.py` and replicated in the online preprocessing for `drive_model.py`. If you change camera resolution or FOV, keep the training‑time and drive‑time preprocessing consistent with how the dataset was generated.