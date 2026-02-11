# CARLA End-to-End Driving – Python API Examples

This folder contains a small end-to-end pipeline for training and running a steering-angle prediction model in the CARLA simulator:

- manual / autopilot data collection from CARLA
- model training (NVIDIA-style CNN)
- feature-map visualization
- autonomous driving using the trained model

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
- `pygame`, `keyboard`
- `matplotlib`

### 1.3. `requirements.txt` and versions

The Conda environment for this project also uses `requirements.txt` in this folder. The key packages and versions you are using are:

- **`future`**: version not pinned (any compatible version).
- **`numpy`**:
  - `numpy==1.18.4` for Python 3 (`python_version >= '3.0'`).
  - Generic `numpy` entry for Python 2 (not relevant for your setup, since you use Python 3).
- **`pygame`**: version not pinned (use a recent version compatible with your OS/Python).
- **`matplotlib`**: version not pinned.
- **`open3d`**: version not pinned.
- **`Pillow`**: version not pinned.

For reproducibility in your thesis, you can mention explicitly that your experiments use **Python 3 + `numpy==1.18.4`** together with the CARLA version corresponding to your simulator installation.

---

## 2. Script overview

### 2.1. `collect_manual.py` – Manual data collection

**Purpose**: Let a human drive a car in CARLA using the keyboard and record episodes of images and controls for training.

- Connects to CARLA on `localhost:2000`.
- Spawns a `model3` vehicle and an RGB camera on the hood.
- Saves data into `dataset_manual/episode_XXX/` folders:
  - images: `frame_id.png`
  - labels: `controls.csv` containing `filename, steer, throttle, brake`.
- Uses a Pygame window for status display.

**Controls**:

- `W` / `S`: throttle / brake
- `A` / `D`: steer left / right
- `SPACE` (hold): start/continue recording **one episode**
- `SPACE` (release): stop recording and write `controls.csv` for that episode
- `ESC`: exit script

**Usage**:

```bash
conda activate carla-gpu
cd path/to/PythonAPI/examples
python collect_manual.py
```

You should see folders like `dataset_manual/episode_000`, `episode_001`, etc.

---

### 2.2. `process_data.py` – Smooth & balance manual dataset

**Purpose**: Take the raw manual dataset in `dataset_manual/` and create a **clean, smoothed, and balanced dataset** in `dataset_processed/` for training.

- Reads each `controls.csv` from `dataset_manual/episode_XXX/`.
- Applies a **moving-average smoothing** to the steering signal (window size `SMOOTHING_WINDOW`, default 5) to transform discrete keyboard inputs into a smoother steering curve.
- Reduces the number of almost-straight frames:
  - If `abs(new_steer) < 0.05`, each frame is kept with probability `KEEP_PROBABILITY` (default 0.3), helping balance turns vs. straight driving.
- Copies only the selected images into `dataset_processed/episode_XXX/` and writes new `controls.csv` files with the **smoothed** steering values.
- At the end, shows histograms comparing the original vs. processed steering-angle distributions.

Key configuration at the top of the file:

- `INPUT_DIR = "dataset_manual"`
- `OUTPUT_DIR = "dataset_processed"`
- `SMOOTHING_WINDOW = 5`
- `KEEP_PROBABILITY = 0.3`

**Usage**:

```bash
conda activate carla-gpu
cd path/to/PythonAPI/examples
python process_data.py
```

This will remove any existing `dataset_processed/` and recreate it from `dataset_manual/`. **You should re-run this script every time you add new episodes to `dataset_manual/`.**

---

### 2.3. `generate_data.py` – Autopilot data collection (optional)

**Purpose**: Collect driving data using CARLA autopilot instead of manual control, useful for a quick dataset or for comparison.

- Connects to CARLA and spawns a `model3` vehicle with an RGB camera.
- Enables `vehicle.set_autopilot(True)` and records images + controls for a fixed duration.
- Saves data into `dataset_small/episode_XXX/` with:
  - images: `frame_id.png`
  - `controls.csv` with `filename, steer, throttle, brake`.
- Can toggle a follow camera using the spectator.

Key parameters at the top of the file:

- `SAVE_FOLDER = "dataset_small"`
- `NUM_EPISODES` – how many episodes to record
- `EPISODE_DURATION` – seconds per episode

**Usage**:

```bash
conda activate carla-gpu
cd path/to/PythonAPI/examples
python generate_data.py
```

---

### 2.4. `train.py` – Train the steering model

**Purpose**: Train a small NVIDIA-style CNN (`SmallNvidiaModel`) to predict steering angles from the **processed** images.

- Uses dataset from `dataset_processed/episode_XXX/` (generated by `process_data.py`):
  - Reads each `controls.csv` file with **smoothed** steering values.
  - Matches image paths with steering values.
- `CarlaDataset`:
  - Applies preprocessing:
    - crop bottom of the image (`crop_img`)
    - convert to YUV color space
    - resize to `66x200`
    - convert to tensor
  - Random horizontal flip augmentation (also flips the steering sign).
- `SmallNvidiaModel`: several convolutional layers followed by fully-connected layers, outputting a single steering value.
- Training parameters (top of file, current defaults):
  - `DATASET_DIR = "dataset_processed"`
  - `BATCH_SIZE = 64`
  - `NUM_EPOCHS = 35`
  - `LEARNING_RATE = 1e-4`
  - Device selection (`cuda` if available, otherwise `cpu`).
- Saves the trained weights as **`model.pth`** in this folder.

**Usage**:

1. Make sure you already have data in `dataset_manual/` (from `collect_manual.py`) and that you have run `process_data.py` so that `dataset_processed/` exists.
2. Run:

```bash
conda activate carla-gpu
cd path/to/PythonAPI/examples
python train.py
```

After training, you should see `model.pth` created.

---

### 2.5. `feature_map.py` – Visualize feature maps

**Purpose**: Visualize the feature maps from the first convolutional layer to understand what the model is learning.

- Loads the **same architecture** (`SmallNvidiaModel`) and weights from `model.pth`.
- Samples a random image from `dataset_processed/`.
- Applies the same preprocessing as in training:
  - crop → YUV → resize → tensor.
- Registers a forward hook on the **first convolutional layer**.
- Runs a forward pass and extracts activations of that layer.
- Displays:
  - the preprocessed input image
  - all feature maps from the first conv layer in a grid using `matplotlib`.

**Usage**:

```bash
conda activate carla-gpu
cd path/to/PythonAPI/examples
python feature_map.py
```

This will open a matplotlib window showing the input and the feature maps.

---

### 2.6. `drive_model.py` – Run autonomous driving

**Purpose**: Use the trained `SmallNvidiaModel` (`model.pth`) to drive the car autonomously in CARLA.

- Loads `model.pth` on GPU if available (`DEVICE = "cuda"` when possible).
- Uses the same preprocessing pipeline as `train.py`:
  - convert CARLA BGRA image → RGB → YUV → crop → resize → tensor.
- Spawns a `model3` vehicle and attaches an RGB camera.
- Continuously:
  - reads the latest camera image
  - runs the model to predict steering
  - applies throttle + predicted steering to the vehicle.
- Spectator camera can follow the car.

**Controls**:

- `V`: toggle spectator camera mode (follow / free).
- `R`: respawn the vehicle at a random spawn point.
- `Ctrl + C`: stop the script.

**Usage**:

1. Make sure `model.pth` exists (trained via `train.py`).
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
   - Launch CARLA and ensure the server is running on `localhost:2000`.

2. **Activate environment and move into examples**

   ```bash
   conda activate carla-gpu
   cd path/to/PythonAPI/examples
   ```

3. **Collect training data (main dataset: `dataset_manual`)**
   - **Option A – Manual driving (primary dataset for your work)**:
     - Run `python collect_manual.py`.
     - Drive with `W/A/S/D`.
     - Hold `SPACE` to record an episode; release `SPACE` to stop and save it.
     - Repeat to gather multiple episodes in `dataset_manual/`.
   - **Option B – Autopilot driving (optional, secondary dataset)**:
     - Run `python generate_data.py`.
     - Let the autopilot drive and record for several episodes into `dataset_small/`.

4. **Process the data into `dataset_processed`**

   ```bash
   conda activate carla-gpu
   cd path/to/PythonAPI/examples
   python process_data.py
   ```

   - This will create/update `dataset_processed/` from your latest manual-driving episodes.

5. **Train the model (on `dataset_processed`)**

   ```bash
   conda activate carla-gpu
   cd path/to/PythonAPI/examples
   python train.py
   ```

   - Wait until training finishes and `model.pth` is saved.

6. **(Optional) Inspect learned features (using `dataset_processed`)**

   ```bash
   conda activate carla-gpu
   cd path/to/PythonAPI/examples
   python feature_map.py
   ```

   - A window will show the input image and the first-layer feature maps.

7. **Run autonomous driving with the trained model**

   ```bash
   conda activate carla-gpu
   cd path/to/PythonAPI/examples
   python drive_model.py
   ```

   - Use `V` to toggle camera follow mode and `R` to respawn the vehicle if needed.

---

## 4. Things you may want to adjust

- **Dataset folders**: your main pipeline is `dataset_manual` → `dataset_processed` → training. `train.py` and `feature_map.py` already point to `dataset_processed`. If you ever want to experiment with the autopilot dataset (`dataset_small` from `generate_data.py`), you can either:
  - change `INPUT_DIR` in `process_data.py` to `dataset_small`, or
  - change `DATASET_DIR` in `train.py` / `DATASET_DIR` in `feature_map.py`
  so they use that alternative dataset. For your main experiments, you keep using the manual-driving data.
- **Training hyperparameters**: adjust `BATCH_SIZE`, `NUM_EPOCHS`, and `LEARNING_RATE` in `train.py` to fit your GPU/CPU performance.
- **Camera resolution / FOV**: the CARLA camera settings are in `collect_manual.py`, `generate_data.py`, and `drive_model.py` and should stay consistent with the preprocessing.

