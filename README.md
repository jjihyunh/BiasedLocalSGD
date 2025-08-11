# Biased Local SGD Framework: Accelerating Deep Learning on Heterogeneous Systems
This repository provides a flexible and modular framework for highly efficient deep learning training on heterogeneous systems. It implements our novel Biased Local SGD approach, which intelligently utilizes diverse compute resources (CPUs, GPUs) by system-aware adjustment of local updates to eliminate synchronization overhead, and by introducing controlled bias in data sampling to enhance convergence. This combined method achieves faster training times with comparable or superior accuracy, making it ideal for cutting-edge distributed deep learning research.

## Software Requirements 
* **Step 1: Make sure you have Python and pip installed on your system.**
* **Step 2: Clone this repository:**
   ```bash
   git clone https://github.com/jjihyunh/BiasedLocalSGD.git
* **Step 3:  Install the required packages:**
    ```bash
   pip install -r requirements.txt

## Instructions
### 1. Training
 1. Set hyper-parameters properly in `config.py`.
    
      *  **General hyperparameters** :  batch_size,  learning rate,  epochs,  decay(epochs for learning rate decay), lr_decay_factor, weight_decay

      *  **Algorithm-specific hyperparameters** : 
      `GPU_interval`: the number of local updates per communication round on fast
      `CPU_interval`: the number of local updates per communication round on slow
      `num_CPU_workers`: the number of CPU workers configured to participate in the distributed training.    These represent the slower compute resources in the heterogeneous system.
      `num_GPU_workers`: the number of GPU workers configured to participate in the distributed training.    These represent the faster compute resources in the heterogeneous system.
      `biased`:
        - **0** → All workers perform uniform random data sampling.
        - **1** → Enable *biased data sampling*, where slower workers prioritize sampling high-loss data.
 2. Run training.
      ```
      mpirun -np 2 python3 main.py
      ```
### 2. Output
This program evaluates the trained model after every epoch and then outputs the results as follows.
 1. `loss.txt`: An output file that contains the average training loss for each epoch.
 2. `acc.txt`: An output file that contains the validation accuracy for each epoch.

## Questions / Comments
 * Jihyun Lim (wlguslim@inha.edu)
 * Sunwoo Lee (sunwool@inha.ac.kr)
