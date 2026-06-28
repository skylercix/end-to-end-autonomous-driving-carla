# CARLA Conditional Imitation Learning for Autonomous Driving

End-to-end deep learning system that drives a car autonomously in the CARLA simulator using only a front-facing RGB camera and a high-level GPS navigation command. Built as my Bachelor's thesis project.

<!-- TODO: replace with a 10-15 second GIF of the car driving autonomously in CARLA -->
![Autonomous driving demo](docs/demo.gif)

---

## Overview

The system learns to drive by imitating recorded demonstrations. A single convolutional neural network takes an RGB image and a navigation command (lane-follow, left, right, straight) and directly predicts steering, throttle, and brake values. The model handles dense urban traffic in CARLA's Town01, including stop-and-go behavior at intersections and following distances behind other vehicles.

## Key Features

- **End-to-end conditional CNN** — single network maps `(image, GPS command)` to `(steer, throttle, brake)`, based on the NVIDIA PilotNet architecture extended with a command branch.
- **Traffic-aware driving** — trained and evaluated with 30–40 NPC vehicles managed by CARLA's `TrafficManager`.
- **Data balancing pipeline** — drops ~90% of straight-driving frames (unless braking is active) to prevent the model from collapsing to a "drive straight" policy.
- **Augmentation and preprocessing** — horizontal flip with command swapping (LEFT ↔ RIGHT), YUV color space conversion, moving-average steering smoothing.
- **Regularization** — Dropout, 15% validation split, ReduceLROnPlateau learning rate scheduler.
- **PID-lite speed controller** — combines model predictions with a proportional speed controller for smoother throttle application.
- **Temporal steering smoothing** — rolling history buffer reduces oscillation at inference time.

## Architecture

<!-- TODO: optional simple diagram of the pipeline -->

```
RGB Camera (200x66, YUV)  ─►  Conv layers (5x NVIDIA-style)  ─┐
                                                              ├─►  FC layers  ─►  [steer, throttle, brake]
GPS command (LANE/L/R/STR) ─►  FC 1→16                        ─┘
```

The image branch produces a 1152-dim feature vector; the command branch produces 16 dims; these are concatenated and passed through fully connected layers (`1168 → 256 → 128 → 64 → 3`).

## Results

<!-- TODO: replace with the actual training_history.png -->
![Training history](docs/training_history.png)

<!-- TODO: optional — first conv layer feature maps -->
![Feature maps of first convolutional layer](docs/feature_maps.png)

The model successfully drives in Town01 with traffic, correctly executing navigation commands at intersections and stopping for vehicles ahead.

## Tech Stack

Python 3.8 · PyTorch · CARLA 0.9.12 · NumPy · Pillow · Pygame · Matplotlib

## Project Structure

```
.
├── collect_autopilot.py   # Data collection in CARLA with manual/autopilot toggle
├── process_data.py        # Smoothing, balancing, cropping, augmentation
├── train.py               # Training loop for the conditional CNN
├── feature_map.py         # Visualization of first conv layer activations
├── drive_model.py         # Autonomous driving with the trained model
```

## Quick Start

```bash
conda activate carla-gpu
cd path/to/PythonAPI/examples

# 1. Collect driving data (hold SPACE to record an episode)
python collect_autopilot.py

# 2. Process and augment the dataset
python process_data.py

# 3. Train the model
python train.py

# 4. Drive autonomously with the trained model
python drive_model.py
```


## About

This project was developed as my Bachelor's thesis in Robotics at Transilvania University of Brașov (2026). The goal was to build a working end-to-end autonomous driving system from data collection to deployment, rather than just train a model on an existing dataset.
