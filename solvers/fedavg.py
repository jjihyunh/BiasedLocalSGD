'''
Jihyun Lim <wlguslim@inha.edu>
Inha University
2025/8/12
'''
from mpi4py import MPI
import torch
from torch import nn
from torch.optim import SGD
from torcheval.metrics import Mean
from torch.optim.lr_scheduler import MultiStepLR
class FedAvg:
    def __init__(self, models, num_classes, **kargs):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.num_classes = num_classes
        self.CPU_interval = kargs["CPU_interval"]
        self.GPU_interval = kargs["GPU_interval"]
        self.num_CPU_workers = kargs["num_CPU_workers"]
        self.num_GPU_workers = kargs["num_GPU_workers"]
        self.learning_rate = kargs["learning_rate"]
        self.decay_epoch = kargs["decay_epoch"]
        self.lr_decay_factor = kargs["lr_decay_factor"]
        self.weight_decay = kargs["weight_decay"]
        self.dataset=kargs["dataset"]
        self.model_name=kargs["model_name"]
        self.num_epochs = kargs["num_epochs"]
        self.local_optimizers = []
        self.local_schdulers = []
        if self.rank==0:
            self.num_local_workers=self.num_CPU_workers
            self.average_interval=self.CPU_interval 
        else:
            self.num_local_workers=self.num_GPU_workers
            self.average_interval=self.GPU_interval 
        for i in range(self.num_local_workers):
            self.local_optimizers.append(SGD(models[i].parameters(),
                                             lr = self.learning_rate,
                                             momentum = 0.9,
                                             weight_decay = self.weight_decay))
            self.local_schdulers.append(MultiStepLR(self.local_optimizers[i],
                                                    milestones = self.decay_epoch,
                                                    gamma = self.lr_decay_factor))
        if self.num_classes == 1:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.device = kargs["device"]

    def round(self, models, dataloaders, local_id):
        model = models[local_id]
        dataloader_iter  = dataloaders[local_id]
        optimizer = self.local_optimizers[local_id]
        loss_mean = Mean().to(self.device)
        loss_mean.reset()
        for i in range(self.average_interval):
            batch = next(dataloader_iter)
            images, labels, indices = batch
            loss,lossmean = self.local_train_step(model, optimizer, images, labels)
            loss_mean.update(lossmean.detach().cpu())
            for j in range (len(indices)):
                self.dataset.score[indices[j]] = loss[j].detach().cpu().numpy()
        return loss_mean

    def local_train_step(self, model, optimizer, data, label):
        model.train()
        data, label = data.to(self.device), label.to(self.device)
        pred = model(data)
        loss = self.loss_fn(pred, label)
        lossmean = loss.mean()
        lossmean.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss,lossmean

    def average_model(self, models, epoch_id):
        # Get the keys.
        keys = []
        myparam = models[0].state_dict()
        for key, tensor in myparam.items():
            keys.append(key)

        # Go through all the parameters one after another.
        new_params = {}
        for key in keys:
            # Collect the tensors from the active local clients.
            local_params = []
            for client_id in range(self.num_local_workers):
                myparams = models[client_id].state_dict()
                local_params.append(myparams[key])

            # Sum up the local tensors and get the number of active local clients.
            num_local_params = len(local_params)
            if num_local_params == 0:
                local_sum = torch.zeros_like(myparams[key])
            else:
                local_sum = torch.sum(torch.stack(local_params), dim = 0)
            num_global_params = self.comm.allreduce(num_local_params, op = MPI.SUM)
            local_sum = local_sum.cpu()

            # Average it across all the processes.
            avg_param = self.comm.allreduce(local_sum, op = MPI.SUM) / num_global_params
            new_params[key] = avg_param
            
        # Apply to all the local models.
        for client_id in range (len(models)):
            myparam = models[client_id].state_dict()
            for key in myparam:
                myparam[key] = new_params[key]
            models[client_id].load_state_dict(myparam)
