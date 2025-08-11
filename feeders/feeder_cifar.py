'''
Jihyun Lim <wlguslim@inha.edu>
Inha University
2025/8/12
'''
import numpy as np
from mpi4py import MPI
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import random
import time
class SubsetWithIndex(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, label = self.dataset[real_idx]
        return image, label, real_idx

class cifar:
    def __init__(self, batch_size,CPU_interval,GPU_interval, num_CPU_workers, num_GPU_workers):
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.rng = np.random.default_rng()
        self.train_batch_size = batch_size
        self.valid_batch_size = 100
        self.num_train_samples = 50000
        self.num_valid_samples = (10000 // self.valid_batch_size) * self.valid_batch_size
        self.CPU_interval = CPU_interval
        self.GPU_interval = GPU_interval
        self.num_CPU_workers = num_CPU_workers
        self.num_GPU_workers = num_GPU_workers
        self.total_workers = num_CPU_workers + num_GPU_workers
        self.training_data = datasets.CIFAR10(root="data",train=True,download=True,
        transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))])
        )
        self.validation_data = datasets.CIFAR10(root="data",train=False,download=True,
        transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))])
        )
        self.num_rounds=int(self.num_train_samples // (self.train_batch_size * self.total_workers * self.GPU_interval))
        self.shuffle_rounds=int(self.num_train_samples //(self.train_batch_size*(self.num_CPU_workers* self.CPU_interval + self.num_GPU_workers * self.GPU_interval)))
        self.num_samples_per_CPU = self.shuffle_rounds* self.CPU_interval* self.train_batch_size
        self.num_samples_per_GPU = self.shuffle_rounds* self.GPU_interval* self.train_batch_size
        self.max_sample_ratio= self.num_train_samples // (self.num_samples_per_CPU *self.num_CPU_workers) 
        self.sample_ratio=4
        self.num_to_sample=int(self.num_samples_per_CPU *self.num_CPU_workers*self.sample_ratio)

        if self.rank==0:
            print( "num rounds in 1epoch: "+str(self.num_rounds))
            print( "shuffle rounds: " + str(self.shuffle_rounds))
            print ("max sampling ratio: " + str(self.max_sample_ratio))
            print ("sampling ratio: " + str(self.sample_ratio))
            print ( "num_samples_per_cpu: "+ str(self.num_samples_per_CPU ))
            print ( "num_samples_per_GPU: "+ str(self.num_samples_per_GPU ))
            print ( "num to samples: " + str(self.num_to_sample))

        self.score = np.full((self.num_train_samples), np.inf)
        self.shuffle()

    def shuffle (self):
        self.shuffled_index = np.arange(self.num_train_samples, dtype='int32')
        random.Random(time.time()).shuffle(self.shuffled_index)
        self.comm.Bcast(self.shuffled_index, root = 0)

    def biased_sample (self):
        candidates = self.rng.choice(np.arange(self.num_train_samples), size = self.num_to_sample, replace = False)
        self.topk_list = candidates[np.argsort(self.score[candidates])[-int(self.num_samples_per_CPU * self.num_CPU_workers):]]
        random.Random(time.time()).shuffle(self.topk_list)
        random.Random(time.time()).shuffle(self.shuffled_index)

    def train_dataset(self, index):
        training_loader = DataLoader(
            SubsetWithIndex(self.training_data, index),
            shuffle=False,
            batch_size=self.train_batch_size,
            drop_last=False
        )
        return training_loader

    def valid_dataset(self):
        validation_loader = DataLoader(
            self.validation_data,
            batch_size=self.valid_batch_size,
            shuffle=False,
            drop_last=False
        )
        return validation_loader
