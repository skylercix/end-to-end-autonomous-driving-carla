## CARLA End-to-End Driving – Python API Examples

This folder contains an end-to-end pipeline for training and running a **navigation-aware steering model** in the CARLA simulator:

- hybrid manual / autopilot data collection with navigation commands
- data processing (smoothing, balancing, cropping, augmentation)
- model training (conditional NVIDIA-style CNN)
- feature-map visualization
- autonomous driving using the trained navigation-aware model

All scripts assume they are run from this `examples` folder so that relative paths to datasets and the model file work correctly.

---

## 1. Environment & Virtual Environment

These examples assume you already have (and **always use**):

- **Conda** installed
- A Conda environment named **`carla-gpu`**
- **CARLA simulator** running and listening on `localhost:2000` (all scripts are hard-coded to this host/port)

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

### 1.3. Exact versions (`requirements.txt`, Python, CARLA)

This project is configured and tested with the following versions:

- **Python**: `3.8.20`
- **CARLA**: `0.9.12` (server and Python API)
- **`future`**: version not pinned (any compatible version).
- **`numpy`**:
  - `numpy==1.18.4` for Python 3 (`python_version >= '3.0'`).
  - Generic `numpy` entry for Python 2 (not relevant here, the code runs on Python 3.8.20).
- **`pygame`**: version not pinned (use a recent version compatible with your OS/Python).
- **`matplotlib`**: version not pinned.
- **`open3d`**: version not pinned.
- **`Pillow`**: version not pinned.

All experiments in this guide assume the environment above.

---

## 2. Script overview

### 2.1. `collect_autopilot.py` – Hybrid data collection with navigation commands

**Purpose**: Collect driving data that includes both **camera images** and **high-level navigation commands**, with the option to switch between CARLA autopilot and manual driving.

- Connects to CARLA on `localhost:2000`.
- Ensures map `Town01` is loaded.
- Spawns a `model3` vehicle and an RGB camera on the hood.
- Uses CARLA’s `BasicAgent` to compute a **navigation route** and corresponding `RoadOption` (LEFT / RIGHT / STRAIGHT / LANE).
- Saves data into `dataset_manual/episode_XXX/` folders:
  - images: `frame_id.png`
  - labels: `controls_nav.csv` containing `filename, steer, throttle, brake, command`, where:
    - `command = 0` → LANE (keep lane / no specific turn)
    - `command = 1` → LEFT
    - `command = 2` → RIGHT
    - `command = 3` → STRAIGHT
- Displays a Pygame window with:
  - current driver mode (AUTO vs MANUAL)
  - recording status
  - GPS command and steering
  - a 2D **radar-style visualization** of the upcoming waypoints (route).

**Driver modes & controls**:

- **Mode toggle**:
  - `M`: toggle between **autopilot** and **manual (WASD)** driving.
- **Manual driving (when manual mode is active)**:
  - `W` / `S`: throttle / brake
  - `A` / `D`: steer left / right (smoothly adjusted over time)
- **Recording episodes**:
  - `SPACE` (hold): start/continue recording **one episode**
  - `SPACE` (release): stop recording and write `controls_nav.csv` for that episode
- **Exit**:
  - `ESC`: exit script

Images are only saved while the vehicle is moving (low-speed frames are discarded).

**Usage**:

```bash
conda activate carla-gpu
cd path/to/PythonAPI/examples
python collect_autopilot.py
```

You should see folders like `dataset_manual/episode_000`, `episode_001`, etc., each with images and a `controls_nav.csv` file including the navigation command.

---

### 2.2. `process_data.py` – Smooth, balance, crop, resize, and augment dataset

**Purpose**: Take the raw navigation-aware dataset in `dataset_manual/` and create a **clean, smoothed, balanced, and structurally preprocessed dataset** in `dataset_processed/` for training.

- Reads each `controls_nav.csv` from `dataset_manual/episode_XXX/`:
  - expects columns: `filename, steer, throttle, brake, command`.
- Applies a **moving-average smoothing** to the steering signal (window size `SMOOTHING_WINDOW`) to transform discrete keyboard/autopilot inputs into a smoother steering curve.
- Reduces the number of almost-straight frames:
  - if `abs(new_steer) < 0.05`, each frame is kept with probability `KEEP_PROBABILITY` (default 0.1), to balance turns vs. straight driving.
- Performs **image-structure preprocessing**:
  - crops out the sky/top of the original `320x240` image (`crop((0, 80, 320, 240))`)
  - resizes the cropped region to `200x66` (width × height)
  - saves only the processed images.
- Performs **data augmentation via horizontal flips**:
  - with probability `FLIP_PROBABILITY` (default 0.5), creates a flipped copy of an image:
    - `steer` is negated
    - `command` is swapped LEFT ↔ RIGHT (1 ↔ 2); STRAIGHT and LANE remain unchanged.
- Writes new `controls_nav.csv` files in `dataset_processed/episode_XXX/` with the updated steering and commands.
- At the end, shows **histograms** comparing:
  - original steering-angle distribution (manual + auto, unsmoothed)
  - processed distribution (smoothed + balanced + flip).

Key configuration at the top of the file:

- `INPUT_DIR = "dataset_manual"`
- `OUTPUT_DIR = "dataset_processed"`
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

This will remove any existing `dataset_processed/` and recreate it from `dataset_manual/`.  
**You should re-run this script every time you add new episodes to `dataset_manual/`.**

---

### 2.3. `train.py` – Train the navigation-aware steering model

**Purpose**: Train a **conditional NVIDIA-style CNN** (`ConditionalNvidiaModel`) that predicts steering angles from preprocessed images **and** the high-level navigation command.

- Uses dataset from `dataset_processed/episode_XXX/` (generated by `process_data.py`):
  - reads each `controls_nav.csv` file with smoothed steering and navigation command.
  - matches image paths with `(steer, command)` pairs.
- `CarlaNavDataset`:
  - iterates over all `episode_XXX` folders in `DATASET_DIR`.
  - for each row in `controls_nav.csv`, loads the **already cropped + resized** image (200×66).
  - converts the image to **YUV/YCbCr** and then to tensor.
  - returns:
    - image tensor (`3×66×200`)
    - command as a scalar float tensor
    - steering as a scalar float tensor.
- `ConditionalNvidiaModel`:
  - image branch: classic NVIDIA CNN stack:
    - conv layers: `(3→24)`, `(24→36)`, `(36→48)`, `(48→64)`, `(64→64)` with strides and ReLUs
    - followed by flatten.
  - command branch:
    - fully-connected `1 → 16` with ReLU.
  - joint head:
    - concatenates image features and command features
    - fully-connected layers `1152+16 → 100 → 50 → 10 → 1` producing a single steering output.
- Training parameters (top of file, current defaults):
  - `DATASET_DIR = "dataset_processed"`
  - `MODEL_SAVE_PATH = "model_nav.pth"`
  - `BATCH_SIZE = 64`
  - `NUM_EPOCHS = 35`
  - `LEARNING_RATE = 1e-4`
  - data loading:
    - `NUM_WORKERS = 8`
    - `PIN_MEMORY`, `PERSISTENT_WORKERS`, `PREFETCH_FACTOR` tuned for GPU.
  - device selection:
    - `DEVICE = "cuda"` if available, otherwise `"cpu"`.
- Uses **MSE loss** on steering and **Adam** optimizer.
- Saves the trained weights as **`model_nav.pth`** in this folder.

**Usage**:

1. Make sure you already have data in `dataset_manual/` (from `collect_autopilot.py`) and that you have run `process_data.py` so that `dataset_processed/` exists.
2. Run:

```bash
conda activate carla-gpu
cd path/to/PythonAPI/examples
python train.py
```

After training, you should see `model_nav.pth` created.

---

### 2.4. `feature_map.py` – Visualize feature maps of the conditional model

**Purpose**: Visualize the feature maps from the **first convolutional layer** of the navigation-aware model to understand what it is learning.

- Loads the same architecture (`ConditionalNvidiaModel`) and weights from `model_nav.pth`.
- Samples a random preprocessed image from `dataset_processed/`.
- Applies the same **YUV + tensor** preprocessing as used in training (no crop/resize here because it was already done in `process_data.py`).
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

### 2.5. `drive_model.py` – Autonomous driving with navigation-aware model

**Purpose**: Use the trained `ConditionalNvidiaModel` (`model_nav.pth`) to **drive the car autonomously** in CARLA, conditioned on the navigation command from `BasicAgent`.

- Loads `model_nav.pth` on GPU if available (`DEVICE = "cuda"` when possible).
- Ensures map `Town01` is loaded.
- Spawns a `model3` vehicle and an RGB camera.
- Uses `BasicAgent` to compute a route and obtain the current `RoadOption` at every step.
- Maps `RoadOption` to a numeric `command`:
  - `0` → LANE
  - `1` → LEFT
  - `2` → RIGHT
  - `3` → STRAIGHT
- Preprocesses camera images:
  - converts CARLA BGRA → RGB
  - converts to YUV/YCbCr
  - resizes to `66×200`
  - converts to tensor and moves to `DEVICE`.
- Maintains a small **steering history buffer** (`STEERING_HISTORY_SIZE`) to smooth predicted steering across frames (temporal smoothing).
- For each new frame:
  - converts image to tensor
  - feeds `(image, command)` to the model
  - smooths the steering over the recent predictions
  - applies a fixed throttle (e.g. `0.4`) and the smoothed steering to the vehicle.
- The Pygame window shows:
  - current model file and device
  - current GPS command and steering
  - controls hint (`[V] Camera | [R] Respawn | [ESC] Exit`)
  - a radar-like 2D view of waypoints around the vehicle (similar visualization as in data collection).
- The spectator camera can follow the car for a third-person view.

**Controls**:

- `V`: toggle spectator camera mode (follow / free).
- `R`: respawn the vehicle at a new random spawn point and regenerate the route.
- `ESC`: stop the script and clean up.

**Usage**:

1. Make sure `model_nav.pth` exists (trained via `train.py`).
2. Ensure CARLA is running.
3. Run:

```bash
conda activate carla-gpu
cd path/to/PythonAPI/examples
python drive_model.py
```

---

## 3. Recommended workflow (step-by-step)

1. **Start CARLA simulator**
   - Launch CARLA and ensure the server is running on `localhost:2000` (map `Town01` will be loaded by the scripts if needed).

2. **Activate environment and move into examples**

   ```bash
   conda activate carla-gpu
   cd path/to/PythonAPI/examples
   ```

3. **Collect navigation-aware training data (main dataset: `dataset_manual`)**
   - Run `python collect_autopilot.py`.
   - Use `M` to toggle between autopilot and manual `W/A/S/D` control.
   - Hold `SPACE` to record an episode; release `SPACE` to stop and save it as `episode_XXX` with `controls_nav.csv`.
   - Repeat to gather multiple episodes in `dataset_manual/` (you can mix manual and autopilot control).

4. **Process the data into `dataset_processed`**

   ```bash
   conda activate carla-gpu
   cd path/to/PythonAPI/examples
   python process_data.py
   ```

   - This will create/update `dataset_processed/` from your latest episodes (smooth, balanced, cropped, resized, and augmented).

5. **Train the navigation-aware model (on `dataset_processed`)**

   ```bash
   conda activate carla-gpu
   cd path/to/PythonAPI/examples
   python train.py
   ```

   - Wait until training finishes and `model_nav.pth` is saved.

6. **(Optional) Inspect learned features (using `dataset_processed`)**

   ```bash
   conda activate carla-gpu
   cd path/to/PythonAPI/examples
   python feature_map.py
   ```

   - A window will show the input image and the first-layer feature maps of the conditional model.

7. **Run autonomous driving with the trained navigation-aware model**

   ```bash
   conda activate carla-gpu
   cd path/to/PythonAPI/examples
   python drive_model.py
   ```

   - Use `V` to toggle camera follow mode and `R` to respawn the vehicle if needed.

---

## 4. Things you may want to adjust

- **Dataset folders**: your main pipeline is `dataset_manual` → `dataset_processed` → training. `train.py` and `feature_map.py` already point to `dataset_processed`.  
  If you ever introduce alternative raw datasets (e.g. separate manual vs autopilot folders), you can:
  - change `INPUT_DIR` in `process_data.py` to the alternative dataset, or
  - change `DATASET_DIR` in `train.py` / `DATASET_DIR` in `feature_map.py`
  so they use that alternative processed dataset.
- **Training hyperparameters**: adjust `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`, `NUM_WORKERS`, and `PREFETCH_FACTOR` in `train.py` to fit your GPU/CPU performance.
- **Camera resolution / FOV / crop**: the CARLA camera settings are in `collect_autopilot.py` and `drive_model.py`. The **structural preprocessing** (crop + resize) is currently centralized in `process_data.py`. If you change camera resolution or FOV, you should keep the training-time and drive-time preprocessing consistent with how the dataset was generated.