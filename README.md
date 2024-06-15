## Overview
This repository contains the source code of VitBit, the CUDA implementation of the Vision Transformer-Base (ViT-Base), and scripts to reproduce the results presented in our paper.

## Prerequisites
- NVIDIA Jetson AGX Orin or similar hardware
- CUDA Toolkit
- Nsight Compute profiling tool

## Installation and Deployment

### Step 1: Clone the Repository
#### Clone the GitHub repository at the commit specified by the DOI and navigate into the cloned directory with your terminal.

### Step 2: Compile the Experimental Environment
#### Compile the experimental environment for all cases:
```sh
 $ cd VitBit/src/
 $ make clean
 $ make
```

### Step 3: Execute Scripts to Reproduce Results
#### Execute the scripts to reproduce the main result, demonstrating the achieved speedup over the baseline, which only utilizes Tensor cores:
```sh
 # ViT-Base with only Tensor cores
 $ ./run_only_tc.sh
 # ViT-Base with Tensor and INT cores
 $ ./run_tc_ic.sh
 # ViT-Base with Tensor and FP cores
 $ ./run_tc_fp.sh
 # ViT-Base with Tensor, INT, and FP cores
 $ ./run_tc_ic_fc.sh
 # ViT-Base with VitBit
 $ ./run_vitbit.sh
```

### Step 4: Execute All Cases for Comprehensive Comparison
#### Execute all cases at once for a comprehensive comparison:
```sh
$ ./run_all_cases.sh
```

### Step 5: Kernel-Level Analysis Using Nsight Compute
#### For kernel-level analysis using Nsight Compute, install the Nsight Compute profiling tool:
```sh
$ cd Profiling_kernels
$ ./profile_only_tc.sh
$ ./profile_tc_ic.sh
$ ./profile_tc_fc.sh
$ ./profile_tc_ic_fc.sh
$ ./profile_vitbit.sh
```
