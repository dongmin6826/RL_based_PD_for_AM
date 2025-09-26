# Reinforcement Learning for Support Structure Minimization in 3D Printing

## Overview
This project explores the application of **deep reinforcement learning** to minimize the support structures required in 3D printing.  
The framework integrates mesh processing, support volume estimation, and a custom OpenAI Gym environment where agents learn cutting and orientation strategies for STL models.  

Although the project did not reach fully optimized results, it reflects an early-stage research attempt during my undergraduate research assistantship.

---

## Repository Structure

config.py # Experiment settings, paths, and hyperparameters
FileHandler.py # Utilities for loading and saving STL/OBJ/3MF files
GYM_wrapper.py # OpenAI Gym interface for the custom environment
main.py # Entry point for training/evaluation with PPO
mesh_processor.py # Mesh slicing and cutting utilities (Trimesh, PyVista)
MeshTweaker.py # Orientation optimization using heuristic methods
PD_environment.py # Environment setup, mesh feature extraction, reward calculation
PD_interface.py # Utility class for orientation and support volume computation
Tweaker.py # CLI entry and tweaking routines (adapted from Tweaker-3)


---

## Key Modules

- **config.py**  
  Defines experiment parameters (number of episodes, action/observation space, model paths, directories).:contentReference[oaicite:0]{index=0}

- **FileHandler.py**  
  Handles reading and writing 3D files in STL/OBJ/3MF formats, supports both ASCII and binary STL parsing.:contentReference[oaicite:1]{index=1}

- **GYM_wrapper.py**  
  Implements an OpenAI Gym-compatible environment wrapper.  
  Defines observation/action spaces, environment resets, and reward logging with TensorBoard.:contentReference[oaicite:2]{index=2}

- **main.py**  
  Training and evaluation script using **Stable-Baselines3 PPO**.  
  Runs episodes, saves models, and exports decomposed STL parts for inspection.:contentReference[oaicite:3]{index=3}

- **mesh_processor.py**  
  Provides mesh slicing and visualization using Trimesh and PyVista libraries.:contentReference[oaicite:4]{index=4}

- **MeshTweaker.py / Tweaker.py**  
  Orientation optimization algorithms adapted from *Tweaker-3*, targeting support volume reduction and printability analysis.:contentReference[oaicite:5]{index=5}:contentReference[oaicite:6]{index=6}

- **PD_environment.py**  
  Defines environment creation, mesh decomposition, reward calculation, and feature extraction (volume, concavity, bounding box, support volume).:contentReference[oaicite:7]{index=7}

- **PD_interface.py**  
  Provides utilities for mesh orientation, rotation, and conversion between different formats. Integrates MeshTweaker with Trimesh for hybrid evaluation.:contentReference[oaicite:8]{index=8}

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run training with PPO
python main.py

# TensorBoard logging (optional)
tensorboard --logdir=./logs







