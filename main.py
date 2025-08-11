'''
Jihyun Lim <wlguslim@inha.edu>
Inha University
2025/8/12
'''
import torch
import config as cfg
from train import framework
from mpi4py import MPI
from solvers.fedavg import FedAvg
from model import ResNet20
from feeders.feeder_cifar import cifar

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    # Dataset
    if cfg.dataset == "cifar10":
        model=cfg.cifar10_config["model"]
        batch_size = cfg.cifar10_config["batch_size"]
        num_epochs = cfg.cifar10_config["epochs"]
        learning_rate = cfg.cifar10_config["learning_rate"]
        num_classes = cfg.cifar10_config["num_classes"]
        decay_epoch = cfg.cifar10_config["decay_epoch"]
        lr_decay_factor = cfg.cifar10_config["lr_decay_factor"]
        weight_decay = cfg.cifar10_config["weight_decay"]
        dataset = cifar(batch_size = batch_size,
                        CPU_interval=cfg.CPU_interval,
                        GPU_interval=cfg.GPU_interval,
                        num_CPU_workers=cfg.num_CPU_workers,
                        num_GPU_workers = cfg.num_GPU_workers
                    )
    else:
        print("config.py has a wrong dataset definition.\n")
        exit()

    if rank==0: # CPU groups
        num_local_workers = cfg.num_CPU_workers
    else:       # GPU groups
        num_local_workers = cfg.num_GPU_workers
        
    # Model
    if model == "ResNet20":
        models = [ResNet20(num_classes).to(device) for _ in range(num_local_workers)]
    else:
        print("Invalid model option!\n")
        exit()

    # Optimizer
    solver = FedAvg(models = models,
                    num_classes = num_classes,
                    CPU_interval=cfg.CPU_interval,
                    GPU_interval=cfg.GPU_interval,
                    num_CPU_workers=cfg.num_CPU_workers,
                    num_GPU_workers = cfg.num_GPU_workers,
                    learning_rate = learning_rate,
                    decay_epoch = decay_epoch,
                    lr_decay_factor = lr_decay_factor,
                    weight_decay = weight_decay,
                    device = device,
                    dataset = dataset,
                    model_name=model,
                    num_epochs=num_epochs)

    if rank == 0:
        print("\n============ Trainer Configuration ============")
        print(f"{'dataset':<24}: {cfg.dataset}")
        print(f"{'number of CPU workers':<24}: {cfg.num_CPU_workers}")
        print(f"{'number of GPU workers':<24}: {cfg.num_GPU_workers}")
        print(f"{'CPU interval':<24}: {cfg.CPU_interval}")
        print(f"{'GPU interval':<24}: {cfg.GPU_interval}")
        print(f"{'Batch size':<24}: {batch_size}")
        print(f"{'training epochs':<24}: {num_epochs}")
        print(f"{'Biased':<24}: {cfg.biased}")
        print("===============================================\n")

    # Trainer
    trainer = framework(models = models,
                        dataset = dataset,
                        solver = solver,
                        num_epochs = num_epochs,
                        num_classes = num_classes,
                        CPU_interval=cfg.CPU_interval,
                        GPU_interval=cfg.GPU_interval,
                        num_CPU_workers=cfg.num_CPU_workers,
                        num_GPU_workers = cfg.num_GPU_workers,
                        biased=cfg.biased,
                        device = device)
    trainer.train()
