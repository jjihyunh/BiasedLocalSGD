'''
Jihyun Lim <wlguslim@inha.edu>
Inha University
2025/8/12
'''
import numpy as np
from mpi4py import MPI
from tqdm import tqdm
import torch
from torcheval.metrics import BinaryAccuracy, MulticlassAccuracy

class framework:
    def __init__(self, models, dataset, solver, **kargs):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.models = models
        self.dataset = dataset
        self.solver = solver
        self.num_epochs = kargs["num_epochs"]
        self.num_classes = kargs["num_classes"]
        self.CPU_interval = kargs["CPU_interval"]
        self.GPU_interval = kargs["GPU_interval"]
        self.num_CPU_workers = kargs["num_CPU_workers"]
        self.num_GPU_workers = kargs["num_GPU_workers"]
        self.biased = kargs["biased"]
        self.device = kargs['device']
        self.total_workers = self.num_CPU_workers + self.num_GPU_workers
        if self.num_classes == 1:
            self.valid_acc = BinaryAccuracy().to(self.device)
        else:
            self.valid_acc = MulticlassAccuracy(num_classes=self.num_classes).to(self.device)
        self.count=0
        
    def train(self):
        # Broadcast the parameters from rank 0 at the first epoch.
        self.broadcast_model()

        for epoch_id in range(self.num_epochs):
            # LR decay.
            for i in range(self.solver.num_local_workers):
                self.solver.local_schdulers[i].step()
                
            for j in tqdm(range(self.dataset.num_rounds), ascii=True): 
             
                # Data shuffle.
                if self.count%self.dataset.shuffle_rounds==0:
                    if self.rank==0:
                        print("============================= Shuffle index =============================")
                    self.count=0
                    
                    if self.biased==0:
                        self.dataset.shuffle()
                        train_dataloaders = []
                        for i in range(self.solver.num_local_workers):
                            start=self.dataset.num_CPU_workers*self.dataset.num_samples_per_CPU 
                            if self.rank==0: # CPU groups
                                index=self.dataset.shuffled_index[i*self.dataset.num_samples_per_CPU : (i+1)*self.dataset.num_samples_per_CPU]
                            else:  # GPU groups
                                index=self.dataset.shuffled_index[start+ i*self.dataset.num_samples_per_GPU : start+(i+1)*self.dataset.num_samples_per_GPU]
                            train_dataloaders.append(iter(self.dataset.train_dataset(index)))
                            
                    elif self.biased==1:
                        # Share loss score.
                        result_array = np.empty_like(self.dataset.score)
                        self.comm.Allreduce(self.dataset.score, result_array, MPI.MIN)
                        self.dataset.score=result_array
                        # Sampling Top-k.
                        self.dataset.biased_sample()
                        train_dataloaders = []
                        for i in range(self.solver.num_local_workers):
                            if self.rank==0: # CPU groups
                                index=self.dataset.topk_list[i*self.dataset.num_samples_per_CPU : (i+1)*self.dataset.num_samples_per_CPU]
                            else:  # GPU groups
                                index=self.dataset.shuffled_index[i*self.dataset.num_samples_per_GPU : (i+1)*self.dataset.num_samples_per_GPU]
                            train_dataloaders.append(iter(self.dataset.train_dataset(index)))
         
                # Training loop.
                local_losses = []
                for local_id in range(self.solver.num_local_workers):
                    local_loss = self.solver.round(self.models, train_dataloaders, local_id)
                    local_losses.append(local_loss.compute().cpu().numpy())
                
                # Aggregate the local models.
                self.solver.average_model(self.models, epoch_id)

                self.count+=1  
                
            # Collect the global training loss.
            global_loss = self.comm.allreduce(sum(local_losses), op = MPI.SUM) / self.total_workers

            # Collect the global validation accuracy.
            local_acc = self.evaluate()
            global_acc = self.comm.allreduce(local_acc, op = MPI.MAX)

            # Logging.
            if self.rank == 0:
                print ("Epoch " + str(epoch_id) +
                        " lr: " + str(self.solver.local_optimizers[0].param_groups[0]['lr']) +
                        " validation acc = " + str(global_acc) +
                        " training loss = " + str(global_loss))
                f = open("acc.txt", "a")
                f.write(str(global_acc) + "\n")
                f.close()
                f = open("loss.txt", "a")
                f.write(str(global_loss) + "\n")
                f.close()


    def evaluate(self):
        valid_dataloader = self.dataset.valid_dataset()
        self.models[0].eval()
        self.valid_acc.reset()
        with torch.no_grad():
            for batch in tqdm(valid_dataloader, ascii=True):
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)
                pred = self.models[0](images)
                pred = torch.argmax(pred, dim=1)
                self.valid_acc.update(pred, labels)
        accuracy = self.valid_acc.compute()
        accuracy = accuracy.cpu().numpy()
        return accuracy

    def broadcast_model(self):
        myparam = self.models[0].state_dict()
        for n, t in myparam.items():
            global_t = self.comm.bcast(t, root = 0)
            myparam[n] = global_t
        self.models[0].load_state_dict(myparam)
        for client_id in range (1, len(self.models)):
            src_param = self.models[0].state_dict()
            dst_param = self.models[client_id].state_dict()
            for key in dst_param:
                dst_param[key] = src_param[key]
            self.models[client_id].load_state_dict(dst_param)
            
