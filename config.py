'''
Jihyun Lim <wlguslim@inha.edu>
Inha University
2025/8/12
'''
cifar10_config = {
    "model" : "ResNet20",
    "batch_size": 64,
    "num_classes": 10,
    "learning_rate": 0.1,
    "epochs": 100,
    "decay_epoch": [60,80],
    "lr_decay_factor": 0.1,
    "weight_decay": 0.0001,
}

dataset = "cifar10"
GPU_interval = 32
CPU_interval = 1
num_GPU_workers = 1
num_CPU_workers = 1
biased= 0
