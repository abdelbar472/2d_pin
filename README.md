# 2D Bin Packing Optimizer

A high-performance 2D bin packing optimizer using a Genetic Algorithm (GA) implemented in C++ with a Python integration layer.

## Features

- **Genetic Algorithm**: Optimizes the order of item placement to minimize the number of planks (bins) required.
- **Ordered Crossover (OX1)**: Ensures valid permutations during evolution.
- **Rotation Support**: Items can be rotated 90 degrees if it improves the packing.
- **Greedy Baseline**: Comparison with First Fit Decreasing (FFD) heuristic.
- **Python Integration**: `pin.py` provides an easy-to-use CLI and visualization tools, leveraging the speed of C++.
- **Flexible Input**: Supports interactive input, JSON files, or default examples.

## Requirements

### C++
- A C++17 compatible compiler (e.g., GCC, Clang, or MSVC).

### Python
- Python 3.6+
- Optional: `matplotlib`, `numpy`, `psutil`, `tqdm` for enhanced visualization and monitoring. (The script will run without them).

## Usage

### 1. Compile the C++ Engine

```bash
g++ -O3 main.cpp -o main.exe
```

### 2. Run the Python Optimizer

The Python script `pin.py` is the main entry point. It automatically detects the compiled C++ executable.

```bash
python3 pin.py
```

### Command Line Arguments

```bash
python3 pin.py --algorithm genetic --no-visualization
python3 pin.py --items items.json --algorithm all
```

## How it Works

1. **Permutation Encoding**: The Genetic Algorithm evolves the *order* in which items are passed to the packing routine.
2. **Sequential Decoder**: The `pack_in_order` routine in C++ places items one-by-one into the first available space in the current or a new plank.
3. **Fitness Function**: Fitness is calculated based on the negative number of planks used (fewer planks = higher fitness).
4. **Hybrid Approach**: Python handles the user interface and high-level logic, while C++ performs the computationally intensive optimization.
