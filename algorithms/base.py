
from utils import *
import copy
import time
from sklearn.metrics import confusion_matrix
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from datasets import Subset4FL, FinetuneSet

shape = {
    'CIFAR10': [3, 32, 32],
    'CIFAR100': [3, 32, 32],
    'MNIST': [1, 28, 28],
    'FashionMNIST': [1, 28, 28],
    'SVHN': [3, 32, 32],
    'STL10': [3, 96, 96],
}
classes = {
    'CIFAR10': 10,
    'CIFAR100': 100,
    'MNIST': 10,
    'FashionMNIST': 10,
    'SVHN': 10,
    'STL10': 10,
}

class ServerBase(object):
    def __init__(self, args, Client) -> None:
        self.algorithm = args.algorithm
        self.agg = args.agg
        self.args = args
        self.best_acc = 0
        self.current_round = 0
        self.client_num = args.client_num
        self.clients = []
        self.dataset = args.dataset
        self.device = torch.device('cuda:0')
        self.exp_tag = args.exp_tag
        self.global_rounds = args.global_rounds
        self.logger = args.logger
        self.load_path = args.load_path
        self.local_steps = args.local_steps
        self.num_join_clients = int(self.client_num * args.join_ratio)
        self.net = args.net
        self.save_dir = args.save_dir
        self.sBN = args.sBN
        self.selected_clients = []
        self.uploaded_models = []
        self.make_dataset(args)
        args.class_num = self.class_num
        self.test_loader = DataLoader(self.testset, batch_size=args.eval_bs, shuffle=False)
        if self.sBN:
            self.batchnorm_dataset = self.make_norm_dataset()
        self.make_client(args, Client)
        self.make_model()
        self.optimizer = get_optimizer(self.global_model, optim_name=args.optim, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_training_steps=self.global_rounds, num_warmup_steps=0)
        self.aggregator = Aggregator(self.agg, self.local_data_num, self.global_model)

    def make_model(self):
        logging.debug('Building models')
        net_builder = get_net_builder(self.net)
        self.global_model = net_builder(self.class_num).to(self.device)
        
    def make_norm_dataset(self):
        logging.debug('Making batchnorm dataset used for sBN')
        return Subset4FL(self.dataset, self.trainset, dataidxs=None, transform=self.testset.transform)

    def make_client(self, args, clientObj):
        logging.debug('Creating clients')
        for i in range(self.client_num):
            trainset = Subset4FL(self.dataset, self.trainset, dataidxs=self.client_data_idx[i])
            client = clientObj(args, i, trainset)
            self.clients.append(client)
        
    def make_dataset(self, args):
        logging.debug('Loading and spliting dataset')
        data_info = fetch_fl_dataset(args.dataset, args.data_dir, args.split_type, args.client_num)
        self.trainset, self.testset = data_info['train_ds'], data_info['test_ds']
        self.client_data_idx = data_info['client_idx']
        self.class_num = data_info['class_num']
        self.local_data_num = data_info['local_data_num']
        self.local_distribution = data_info['local_distribution']
        if self.args.ft:
            trainset = FinetuneSet(self.dataset, self.trainset, self.args.ft_data_per_cls)
            self.train_loader = DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True)

    def select_clients(self):
        selected_clients = list(np.random.choice(self.clients, self.num_join_clients, replace=False))
        return selected_clients
    
    def receive_messages(self, round_idx):
        """
        collect local models
        """
        raise NotImplementedError
    
    def train(self, round):
        st = time.time()
        self.global_model.train(True)
        ce_loss = nn.CrossEntropyLoss()
        clip_grad = self.args.clip_grad
        for epoch in range(self.local_steps):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.global_model(x)
                loss = ce_loss(logits, y)
                loss.backward()
                if clip_grad > 0:
                    clip_grad_norm_(self.global_model.parameters(), clip_grad)
                self.optimizer.step()
        print(f"server_train_time>> {(time.time() - st) / 60:.2f} min")
            
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

            self.receive_messages(round_idx)
            self.aggregator.aggregate(self.uploaded_models)
            if self.args.ft:
                self.train(round_idx)
            torch.cuda.empty_cache()
            self.scheduler.step()
            if self.sBN:
                make_batchnorm_stats(self.batchnorm_dataset, self.global_model, self.device)
            self.test(round_idx)
            print(f'time cost {(time.time() - st_time)/60:.2f} min')
    

    @torch.no_grad()
    def test(self, round_idx):
        logging.info('Testing at round {}/{}'.format(round_idx, self.global_rounds))
        st = time.time()
        loader = self.test_loader
        all_y, all_logits = [], []
        self.global_model.train(False)
        self.global_model.to(self.device)
        for x, y in loader:   
            x, y = x.to(self.device), y.to(self.device)
            logits = self.global_model(x)
            all_y.append(y)
            all_logits.append(logits)
        y = torch.cat(all_y, dim=0)
        logits = torch.cat(all_logits, dim=0)
        test_acc = (y == torch.argmax(logits, dim=1)).float().mean().item() * 100

        cm = confusion_matrix(y.cpu().numpy(), torch.argmax(logits, dim=1).cpu().numpy(), labels=range(self.class_num))
        class_per_acc = cm.diagonal() / (cm.sum(axis=0) + 1e-8)
        logs = '------ precision: '
        for i, acc in enumerate(class_per_acc):
            logs += f'{acc * 100:.2f} '
        logging.info(logs)

        recall_per_class = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
        logs = '------ recall:    '
        for i, acc in enumerate(recall_per_class):
            logs += f'{acc * 100:.2f} '
        logging.info(logs)
        best = False
        if test_acc > self.best_acc:
            self.best_acc = test_acc
            best = True
        logging.info(f'------ test acc: {test_acc:.2f}% & best acc: {self.best_acc:.2f}%')
        log_dict = {'test-top1_acc': test_acc, 'test-best_acc': self.best_acc}
        self.logger.log(log_dict, step=round_idx)
        self.save_model(round_idx, best=best)
        logging.info(f"server_test_time>> {(time.time() - st) / 60:.2f} min")


    def save_model(self, round, best=False):
        save_path = os.path.join(self.save_dir, self.exp_tag)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if best:
            path = os.path.join(save_path, f'{self.net}_models_best.pth')
            ckpt = {
                'ckpt_model': self.global_model.state_dict(),
            }
            torch.save(ckpt, path)
        else:
            path = os.path.join(save_path, f'{self.net}_{round}.pth')
            ckpt = {
                'ckpt_model': self.global_model.state_dict(),
            }
            torch.save(ckpt, path) 
        

class ClientBase(object):
    def __init__(self, args, id, trainset) -> None:
        self.id = id
        self.net = args.net
        self.load_path = args.load_path
        self.local_steps = args.local_steps
        self.trainset = trainset
        self.class_num = args.class_num
        self.batch_size = args.c_batch_size
        self.logger = args.logger
        self.exp_tag = args.exp_tag
        self.device = torch.device('cuda:0')
        self.make_model()
        self.optimizer = get_optimizer(self.model, optim_name=args.optim, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        self.optimizer_dict = self.optimizer.state_dict()

    def make_model(self):
        logging.debug('Building models')
        net_builder = get_net_builder(self.net)
        self.model = net_builder(self.class_num).to(self.device)

    def set_parameters(self, state_dict):
        self.model.load_state_dict(state_dict)
    
    def prepare(self, lr, state_dict):
        self.model.to(self.device)
        self.set_parameters(state_dict)
        for group in self.optimizer_dict['param_groups']:
            group['lr'] = lr
        self.optimizer.load_state_dict(self.optimizer_dict)

    def train(self, round_idx, lr, model_dict):
        pass


class Aggregator(object):
    def __init__(self, agg, client_data_num, global_model) -> None:
        self.device = torch.device('cuda:0')
        self.agg = agg
        self.client_data_num = client_data_num
        self.global_model = global_model
        self.global_optimizer = get_optimizer(global_model, optim_name='SGD', lr=1, momentum=0.5, weight_decay=0, nesterov=False)

    
    def aggregate(self, uploaded_models):
        logging.debug('Performing aggregation')
        st = time.time()
        
        weight = []
        if self.agg == 'average':
            weight = [1 / len(uploaded_models)] * len(uploaded_models)
        elif self.agg == 'weighted_average':
            for i in uploaded_models.keys():
                weight.append(self.client_data_num[i])
            weight = [w / sum(weight) for w in weight]
        models = []
        for i in uploaded_models.keys():
            models.append(uploaded_models[i].state_dict())
        with torch.no_grad():
            
            shadow_dict = models[0]
            for i in range(len(models)):
                w = weight[i]
                model = models[i]
                if i == 0:
                    for key in shadow_dict.keys():
                        shadow_dict[key] = model[key] * w
                else:
                    for key in shadow_dict.keys():
                        shadow_dict[key] += model[key] * w
            self.global_model.load_state_dict(shadow_dict)
            # self.global_optimizer.zero_grad()
            # for k, v in self.global_model.state_dict().items():
            #     if 'weight' in k or 'bias' in k:
            #         continue
                
            #     v.data = shadow_dict[k].data.clone()

            # for name, param in self.global_model.named_parameters():
            #     param.grad = (param.data - shadow_dict[name].data).detach()

            # self.global_optimizer.step()

        logging.info(f'Time cost of aggregation: {(time.time() - st) / 60:.2f} min')   