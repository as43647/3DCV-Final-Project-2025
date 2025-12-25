# 3DCV-Final-Project-2025

# Occlusion-Aware Volumetric Video Streaming for Mixed Reality

A comprehensive pipeline for bandwidth-efficient volumetric video streaming on Mixed Reality headsets, implementing gaze-contingent foveated rendering with elliptical log-polar transformation and differential frame streaming.

## 📁 Repository Structure

```
├── elliptic.cu                   # CUDA implementation with elliptical log-polar transformation
├── 3dcv_final.cu                 # CUDA implementation with standard circular log-polar transformation
├── diffrential_sequence.py       # Python script for differential frame streaming
├── longdress_vox10_1051.ply      # Sample point cloud frame 1 (MPEG 8i dataset)
├── longdress_vox10_1052.ply      # Sample point cloud frame 2 (MPEG 8i dataset)
├── final_report_group_12.pdf     # Complete project report
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites
- **NVIDIA GPU** with CUDA support (Compute Capability 6.0+)
- **CUDA Toolkit** 11.0 or higher
- **Python 3.8+** with numpy and matplotlib
- **g++** compiler

### Installation

```bash
# Clone repository
git clone <repository-url>
cd <repository-directory>

# Compile CUDA programs
nvcc -o theia_elliptic elliptic.cu -O2 -arch=sm_86
nvcc -o theia_standard 3dcv_final.cu -O2 -arch=sm_86

# Install Python dependencies
pip install numpy matplotlib
```

## 📊 Three Optimization Strategies

### 1️⃣ Occlusion-Aware Point Culling
- **Implementation**: `elliptic.cu` & `3dcv_final.cu`
- **Purpose**: Removes hidden points not visible from user's viewpoint
- **Result**: ~31-34:1 compression ratio

### 2️⃣ Foveated Streaming (HVS-Informed)
- **Elliptical Transformation**: `elliptic.cu` (1.6:1 horizontal-to-vertical compression)
- **Circular Transformation**: `3dcv_final.cu` (uniform compression)
- **Result**: 33.5:1 vs 31.4:1 compression ratio (6.7% improvement)

### 3️⃣ Differential Frame Streaming
- **Implementation**: `diffrential_sequence.py`
- **Purpose**: Sends only changed points between consecutive frames
- **Benefit**: Further bandwidth reduction based on temporal coherence

## 🔧 Usage Guide

### A. Elliptical Log-Polar Transformation

```bash
./theia_elliptic [input_file] [output_file] [gaze_pos_x] [gaze_pos_y] [gaze_pos_z] 
                 [gaze_dir_x] [gaze_dir_y] [gaze_dir_z]
```

**Examples:**
```bash
# Use defaults (a=1.6, b=1.0 elliptical compression)
./theia_elliptic

# Custom parameters
./theia_elliptic longdress_vox10_1051.ply output.ply 0.0 1.5 -1.0 0.0 0.0 -1.0

# Help
./theia_elliptic --help
```

**Output**: `theia_elliptical_output.ply` with ~23K points (33.5:1 compression)

### B. Standard Circular Transformation

```bash
./theia_standard [input_file] [output_file] [gaze_pos_x] [gaze_pos_y] [gaze_pos_z] 
                 [gaze_dir_x] [gaze_dir_y] [gaze_dir_z]
```

**Examples:**
```bash
# Use defaults (uniform circular compression)
./theia_standard

# Custom gaze
./theia_standard longdress_vox10_1051.ply output.ply 0.5 1.8 -0.8

# Help
./theia_standard --help
```

**Output**: `theia_clean_output.ply` with ~24K points (31.4:1 compression)

### C. Differential Frame Analysis

```bash
python diffrential_sequence.py <file1.ply> <file2.ply> [OPTIONS]
```

**Examples:**
```bash
# Basic comparison
./theia_standard longdress_vox10_1051.ply output1051.ply
./theia_standard longdress_vox10_1052.ply output1052.ply
python diffrential_sequence.py output1051.ply output1051.ply

# Custom threshold and output
python diffrential_sequence.py frame1.ply frame2.ply -o diff.ply -t 0.005

# Help
python diffrential_sequence.py --help
```

**Output**: 
- `differences.ply` - Points that changed beyond threshold
- `differences_stats.txt` - Statistical analysis
- `differences_histogram.png` - Distance distribution

## 📈 Performance Comparison

| Metric | Circular (3dcv_final) | Elliptical (elliptic) | Improvement |
|--------|----------------------|----------------------|-------------|
| **Compression Ratio** | 31.4:1 | 33.5:1 | +6.7% |
| **Output Points** | 24,723 | 23,124 | -6.5% |
| **Processing Time** | 10 ms | 9 ms | -10% |
| **Bandwidth Saving** | Baseline | Additional 6.5% | Significant |

## 🧪 Complete Workflow Example

```bash
# Step 1: Process frame 1 with elliptical transformation
./theia_elliptic longdress_vox10_1051.ply frame1_processed.ply

# Step 2: Process frame 2 with elliptical transformation  
./theia_elliptic longdress_vox10_1052.ply frame2_processed.ply

# Step 3: Analyze differences between frames
python diffrential_sequence.py frame1_processed.ply frame2_processed.ply -t 0.002 -o frame_diff.ply

# Step 4: Compare with circular transformation
./theia_standard longdress_vox10_1051.ply frame1_circular.ply
./theia_standard longdress_vox10_1052.ply frame2_circular.ply
```

## 🔍 PLY File Requirements

All programs require PLY files in ASCII format with:
```
ply
format ascii 1.0
element vertex [N]
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
```

## ⚙️ Customization Parameters

### Elliptical Transformation (elliptic.cu)
```cpp
const float ELLIPTICAL_A = 1.6f;  // Horizontal compression
const float ELLIPTICAL_B = 1.0f;  // Vertical compression
```

### Processing Parameters
```cpp
float rate_adapt = 1.0f;     // Adaptation rate
bool augmentation = false;    // Inpainting augmentation
```

### Differential Analysis
```python
THRESHOLD = 0.001  # Distance threshold for detecting differences
```

## 📊 Expected Results

### Sample Output (longdress_vox10_1051.ply):
```
=== THEIA ELLIPTICAL LOG-POLAR PROCESSING ===
Elliptical parameters: a=1.6 (horizontal), b=1.0 (vertical)
Read 775,745 points
Number of selected points: 23,124
Processing time: 9 ms
Compression ratio: 33.5:1
```

### Differential Analysis Output:
```
Total points: 775,745
Points beyond threshold: 15,328 (1.98%)
Mean distance: 0.00042
Max distance: 0.0125
```

## 🐛 Troubleshooting

### CUDA Compilation Errors
```bash
# For RTX 30xx series
nvcc -arch=sm_86

# For RTX 20xx series  
nvcc -arch=sm_75

# For GTX 10xx series
nvcc -arch=sm_61
```

### File Format Errors
- Ensure PLY files are ASCII format
- Check for proper header structure
- Verify color properties exist

### Python Dependencies
```bash
pip install numpy matplotlib
```

## 📚 Documentation Files

- **`final_report_group_12.pdf`**: Complete project report with methodology, results, and analysis
- **Source code comments**: Detailed usage instructions in each file header
- **This README**: Quick start guide and examples

## 🎯 Key Features

✅ **Real-time Processing**: <20ms per frame on RTX 3070  
✅ **High Compression**: 31-34:1 compression ratios  
✅ **Perceptual Quality**: HVS-informed transformation  
✅ **Bandwidth Reduction**: 6.5% improvement with elliptical model  
✅ **Easy Integration**: Command-line interface with sensible defaults  

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{theia2025,
  title={Occlusion-Aware Volumetric Video Streaming for Bandwidth-Efficient 3D Viewing},
  author={Kamrani, Mohammadreza and Wu, Zhonghan},
  year={2025},
  booktitle={3DCV Course Project}
}
```

## 👥 Authors

- **Mohammadreza Kamrani** (D13949003)
- **吳忠翰** (M11207327)

3DCV Course Project, December 2025

## 📄 License

Educational and research use only. See individual file headers for details.

---

**💡 Tip**: Start with `./theia_elliptic` for best compression results, then use `diffrential_sequence.py` to analyze temporal differences between frames.
