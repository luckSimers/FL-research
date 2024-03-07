# This file contains the implementation of FedAvg algorithm
import copy
import time
import logging
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import *
from algorithms import ClientBase, ServerBase


class FedAvg(ServerBase):
    def __init__(self, args) -> None:
        super(FedAvg, self).__init__(args, FedAvgClient)
    
    def receive_messages(self):
        """
        collect models, training samples
        """
        logging.info('Receiving models from clients')
        self.uploaded_models = {}
        for i, client in enumerate(self.selected_clients):
            id = client.id
            model = copy.deepcopy(client.model).to(self.device)
            self.uploaded_models[id] = model
            
    def run(self):
        """
        in this function, server only:
            1.send model to clients;
            2.select active clients for local training;
            3.receive models from clients;
            4.aggregate models;
        rewrite this function if you wanna use a more complex algorithm
        """
        for round_idx in range(self.global_rounds):
            print(f'--------------{round_idx}-----------------')
            st_time = time.time()
            self.selected_clients = self.select_clients()
            lr = self.scheduler.get_last_lr()[0]
            model_dict = self.global_model.state_dict()
            for i, client in enumerate(self.selected_clients):
               client.train(round_idx, lr, model_dict)

            self.receive_messages()
            self.aggregator.aggregate(self.uploaded_models)
            torch.cuda.empty_cache()
            self.scheduler.step()
            if self.sBN:
                make_batchnorm_stats(self.batchnorm_dataset, self.global_model, self.device)
            self.test(round_idx)
            print(f'time cost {(time.time() - st_time)/60:.2f} min')
    
    @staticmethod
    def get_argument():
        return [
            Special_Argument('--sBN', type=bool, default=False, help='use sBN or not'),
        ]


def plot_tsne(model, dataloader, device, use_prototype=False):
    model.eval()
    all_reps = []
    all_y = []
    for data in dataloader:
        x, y = data['x_lb'].to(device), data['y_lb'].to(device)
        with torch.no_grad():
            reps = model.base(x)
            all_reps.append(reps)
            all_y.append(y)
    reps = torch.cat(all_reps, dim=0)
    y = torch.cat(all_y, dim=0)

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(reps.cpu().numpy())
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y.cpu().numpy(), s=10, cmap='Set1_r')
    plt.colorbar(ticks=range(3))
    plt.savefig(f'{use_prototype}_tsne.png')

class FedAvgClient(ClientBase):
    def __init__(self, args, id, trainset) -> None:
        super(FedAvgClient, self).__init__(args, id, trainset)
        self.clip_grad = args.clip_grad
        self.loss = nn.CrossEntropyLoss()

    def train(self, round_idx, lr, state_dict):
        st = time.time()
        self.prepare(lr, state_dict)
        loader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        self.model.train(True)
        for step in range(self.local_steps):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.loss(logits, y)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.optimizer.step()
                
        print(f'C{self.id:<2d}>> training cost {(time.time() - st)/60:.2f} min')
        self.model.to('cpu')
