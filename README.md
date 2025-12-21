# 3DCV-Final-Project-2025
Implementation of occlusion-aware volumetric video streaming using **Log-Polar transformation** and **Implicit HPR** (raw CUDA kernels).

## Overview
This project implements an occlusion-aware pipeline for voxelized volumetric video (8i Voxelized Full Bodies).
The core components include:
- Log-Polar transformation for view-dependent processing
- Implicit HPR (Hidden Point Removal) for occlusion handling
- CUDA kernel implementation (no external dependencies)

## System Requirements
- **OS:** Ubuntu 20.04 / 22.04 (recommended), or Windows (see notes below)
- **GPU:** NVIDIA GPU (Compute Capability 6.0+)
- **Toolchain:** NVIDIA CUDA Toolkit (`nvcc`) + C++ build essentials

## Dataset
This project uses the **8i Voxelized Full Bodies (MPEG)** dataset (e.g., *Loot*, *LongDress*).

- Loot: https://drive.google.com/drive/folders/1fbDX3Je0em2fG4_hTaq08lZ6nx-5Y868?usp=drive_link  
- LongDress: https://drive.google.com/drive/folders/117AKgjtffmNUeM9t20B5HeFBjnd7OMl3?usp=drive_link

### Expected file naming (important)
The current implementation expects a **PLY file prefix** like:
- `.../loot/Ply/loot_vox10_`
- `.../longdress/Ply/longdress_vox10_`

Your frames should follow a pattern similar to:
- `loot_vox10_0000.ply`, `loot_vox10_0001.ply`, ...
- `longdress_vox10_0000.ply`, `longdress_vox10_0001.ply`, ...

If your dataset naming differs, update the filename parsing/loading logic in `main.cu`.

## Prerequisites & Installation
No external libraries are required. Only the **CUDA Toolkit** and standard build tools.

### 1) Check CUDA Toolkit
```bash
nvcc --version
```
### 2) Install build tools (Ubuntu)
```bash
sudo apt update
sudo apt install -y build-essential git
```
### 3) Bulid
For RTX 30 Series (Ampere):
```bash
nvcc -o hpr_result main.cu -O3 -arch=sm_86
```
For RTX 40 Series (Ada Lovelace):
```bash
nvcc -o hpr_result main.cu -O3 -arch=sm_89
```
### 4) Run
Make a output folder:
```bash
mkdir -p output
```
Loot (change startFrame = 1000):
```bash
./hpr_result /your_path/loot/loot/Ply/loot_vox10_
```
LongDress (change startFrame = 1051):
```bash
./hpr_result /your_path/longdress/longdress/Ply/longdress_vox10_
```
