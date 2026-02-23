# Biased Local SGD Framework: Accelerating Deep Learning on Heterogeneous Systems
This repository provides a flexible framework for highly efficient deep learning training on heterogeneous systems. It implements our novel Biased Local SGD approach, which intelligently utilizes diverse compute resources (CPUs, GPUs) by system-aware adjustment of local updates to eliminate synchronization overhead, and by introducing controlled bias to enhance convergence. This combined method achieves faster training times, while achieving comparable or even superior model accuracy within the same time budget. This framework empowers researchers to fully leverage their heterogeneous hardware for deep learning.

## ⚙️ Software Requirements 
> This code was developed and tested with  PyTorch 2.7.1 and tensorflow 2.15.0.

* **Step 1: Make sure you have `conda` installed on your system.**
* **Step 2: Create a new conda environment.**
   ```bash
   conda create -n myenv python=3.12.7
* **Step 3: Activate the environment.**
   ```bash
   conda activate myenv
* **Step 4: Clone this repository.**
   ```bash
   git clone https://github.com/jjihyunh/BiasedLocalSGD.git
   cd BiasedLocalSGD
* **Step 5: Install required packages.**
   ```bash
   conda install -c conda-forge mpi4py=4.1.1
   conda install -c conda-forge openai
   pip install -r requirements.txt

## 📝 Instructions
### 1. Training
* **Step 1: Set hyper-parameters properly in `config.py`.**

   *  **General hyperparameters** :  batch_size,   learning rate,   epochs,   decay_epoch(epochs for learning rate decay),  lr_decay_factor,   weight_decay
   *  **Algorithm-specific hyperparameters** : 
      
         `GPU_interval`: the number of local updates per communication round on fast

         `CPU_interval`: the number of local updates per communication round on slow
       
         `num_CPU_workers`: the number of CPU workers configured to participate in the distributed training. These represent the slower compute resources in the heterogeneous system.
       
         `num_GPU_workers`: the number of GPU workers configured to participate in the distributed training. These represent the faster compute resources in the heterogeneous system.
       
         `biased`:
      - **0** → All workers perform uniform random data sampling.
      - **1** → Enable *biased data sampling*, where slower workers prioritize sampling high-loss data.
    
* **Step 2: Run training.**
     ```bash
      mpirun -np 2 python3 main.py

### 2. Output
This program evaluates the trained model after every epoch and then outputs the results as follows.
 * `loss.txt`: An output file that contains the average training loss for each epoch.
 * `acc.txt`: An output file that contains the validation accuracy for each epoch.

## 📞 Questions / Comments
 * Jihyun Lim (wlguslim@inha.edu)
 * Sunwoo Lee (sunwool@inha.ac.kr)
