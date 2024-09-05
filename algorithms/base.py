import copy
import time
import torch.cuda as cuda
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from sklearn.metrics import confusion_matrix
from datasets import Subset4FL
from utils import *


class ServerBase(object):
    def __init__(self, args, Client) -> None:
        self.algorithm = args.algorithm
        self.args = args
        self.best_acc = 0
        self.current_round = 0
        self.num_classes = args.num_classes
        self.client_num = args.client_num
        self.clients = []
        self.dataset = args.dataset
        self.device = args.device
        self.exp_tag = args.exp_tag
        self.global_rounds = args.global_rounds
        self.logger = args.logger
        self.printer = args.printer
        self.load_path = args.load_path
        self.active_client = int(self.client_num * args.join_ratio)
        self.net = args.net
        self.save_dir = args.save_dir
        self.sBN = args.sBN
        self.selected_clients = []
        self.uploaded_models = []
        self.uploaded_weights = []
        self.ft = False
        self.make_dataset(args)
        self.make_client(args, Client)
        self.make_model()
        self.make_optimizer(args)
        if self.sBN:
            self.batchnorm_dataset = self.make_norm_dataset()
            

    def make_optimizer(self, args):
        self.optimizer = get_optimizer(self.global_model, optim_name=args.optim, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_training_steps=self.global_rounds, num_warmup_steps=0)
        
    def make_model(self):
        self.printer.debug('Building models')
        net_builder = get_net_builder(self.net)
        self.global_model = net_builder(self.num_classes).to(self.device)
        
    def make_norm_dataset(self):
        self.printer.debug('Making batchnorm dataset used for sBN')
        return Subset4FL(self.dataset, self.trainset, dataidxs=None, transform=self.testset.transform)

    def make_client(self, args, clientObj):
        self.printer.debug('Creating clients')
        for i in range(self.client_num):
            trainset = Subset4FL(self.dataset, self.trainset, dataidxs=self.client_data_idx[i])
            client = clientObj(args, i, trainset)
            self.clients.append(client)
        
    def make_dataset(self, args):
        self.printer.debug('Loading and spliting dataset')
        data_info = fetch_fl_dataset(args.dataset, args.data_dir, args.split_type, args.client_num, args.ft_data_per_cls)
        self.trainset, self.testset = data_info['train_ds'], data_info['test_ds']
        self.client_data_idx = data_info['client_idx']
        self.local_data_num = data_info['local_data_num']
        self.local_distribution = data_info['local_distribution']
        self.test_loader = DataLoader(self.testset, batch_size=args.eval_bs, shuffle=False)
        if args.ft_data_per_cls > 0:
            self.ft = True
            self.local_steps = args.s_steps
            self.batch_size = 20 if args.ft_data_per_cls * self.num_classes < 1000 else 200
            self.ft_idx = data_info['ft_idx']
            ft_set = Subset4FL(
                self.dataset, self.trainset, self.ft_idx,
                self.testset.transform,
                compress_freq=args.compress_freq,
                randomize=args.randomize,
                del_num=args.del_freq_chs
            )
            self.ft_loader = DataLoader(ft_set, batch_size=self.batch_size, shuffle=True)
            
    def select_clients(self):
        selected_clients = list(np.random.choice(self.clients, self.active_client, replace=False))
        return selected_clients
    
    def receive_messages(self):
        """
        collect models, training samples
        """
        self.printer.info('Receiving models from clients')
        self.uploaded_models = []
        self.uploaded_weights = []
        for i, client in enumerate(self.selected_clients):
            id = client.id
            model = copy.deepcopy(client.model).to(self.device).state_dict()
            self.uploaded_models.append(model)
            self.uploaded_weights.append(self.local_data_num[id])
    
    @torch.no_grad()
    def aggregate(self):
        self.printer.debug('Performing aggregation')
        st = time.time()
        
        sample_num = sum(self.uploaded_weights)
        weights = [w / sample_num for w in self.uploaded_weights]

        if len(self.uploaded_models) > 0:
            self.printer.info('--------------start aggregation--------------------')
            self.printer.info(f'agg weights: {weights}')
            new_dict = self.global_model.state_dict()
            for i in range(len(weights)):
                w = weights[i]
                if i == 0:
                    for v in new_dict.values():
                        v = (v * w).type(v.dtype)
                else:
                    for v, lv in zip(new_dict.values(), self.uploaded_models[i].values()):
                        v += (w * lv).type(v.dtype)
        else:
            self.printer.info('no uploaded models, skip aggregation')

        self.printer.info(f'Time cost of aggregation: {(time.time() - st) / 60:.2f} min')  
    
    def train(self, round):
        st = time.time()
        self.global_model.train(True)
        ce_loss = nn.CrossEntropyLoss()
        clip_grad = self.args.clip_grad
        for epoch in range(self.local_steps):
            for x, y in self.ft_loader:
                x, y = x.to(self.device), y.to(self.device)
                if len(y) < 2:
                    continue
                self.optimizer.zero_grad()
                logits = self.global_model(x)
                loss = ce_loss(logits, y)
                loss.backward()
                if clip_grad > 0:
                    clip_grad_norm_(self.global_model.parameters(), clip_grad)
                self.optimizer.step()
        self.printer.info(f"server_train_time>> {(time.time() - st) / 60:.2f} min")
            
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
            self.printer.info(f'Round {round_idx}/{self.global_rounds}...')
            st_time = time.time()
            self.selected_clients = self.select_clients()
            lr = self.scheduler.get_last_lr()[0]
            
            for i, client in enumerate(self.selected_clients):
               client.train(round_idx, lr, self.global_model)

            self.receive_messages()
            self.aggregate()
            if self.ft:
                self.train(round_idx)
            cuda.empty_cache()
            self.scheduler.step()
            if self.sBN:
                make_batchnorm_stats(self.batchnorm_dataset, self.global_model, self.device)
            self.test(round_idx)
            self.printer.info(f'time cost {(time.time() - st_time)/60:.2f} min')
    

    @torch.no_grad()
    def test(self, round_idx):
        self.printer.info('Testing at round {}/{}'.format(round_idx, self.global_rounds))
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

        cm = confusion_matrix(y.cpu().numpy(), torch.argmax(logits, dim=1).cpu().numpy(), labels=range(self.num_classes))
        class_per_acc = cm.diagonal() / (cm.sum(axis=0) + 1e-8)
        logs = '------ precision: '
        for i, acc in enumerate(class_per_acc):
            logs += f'{acc * 100:.2f} '
        self.printer.info(logs)

        recall_per_class = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
        logs = '------ recall:    '
        for i, acc in enumerate(recall_per_class):
            logs += f'{acc * 100:.2f} '
        self.printer.info(logs)
        best = False
        if test_acc > self.best_acc:
            self.best_acc = test_acc
            best = True
        self.printer.info(f'------ test acc: {test_acc:.2f}% & best acc: {self.best_acc:.2f}%')
        log_dict = {'test-top1_acc': test_acc, 'test-best_acc': self.best_acc}
        self.logger.log(log_dict, step=round_idx)
        if best:
            self.save_model(round_idx, best=best)
        self.printer.info(f"server_test_time>> {(time.time() - st) / 60:.2f} min")


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
        

class ClientBase(object):
    def __init__(self, args, id, trainset) -> None:
        self.id = id
        self.net = args.net
        self.load_path = args.load_path
        self.local_steps = args.c_steps
        self.num_classes = args.num_classes
        self.batch_size = args.c_bs
        self.logger = args.logger
        self.printer = args.printer
        self.exp_tag = args.exp_tag
        self.device = args.device
        self.clip_grad = args.clip_grad
        self.trainset = trainset
        self.loader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        self.make_model()
        self.make_optimizer(args)

    def make_model(self):
        self.printer.debug('Building models')
        net_builder = get_net_builder(self.net)
        self.model = net_builder(self.num_classes).to(self.device)
    
    def make_optimizer(self, args):
        self.optimizer = get_optimizer(self.model, optim_name=args.optim, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        self.optimizer_dict = self.optimizer.state_dict()

    def set_parameters(self, model):
        for p, new_p in zip(self.model.parameters(), model.parameters()):
            p.data = new_p.data.clone()
    
    def prepare(self, lr, model):
        self.model.to(self.device)
        self.set_parameters(model)
        for group in self.optimizer_dict['param_groups']:
            group['lr'] = lr
        self.optimizer.load_state_dict(self.optimizer_dict)

    def train(self, round_idx, lr, model_dict):
        pass

    @torch.no_grad()
    def test(self):
        self.model.train(False)
        all_y, all_logits = [], []
        for x, y in self.loader:   
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            all_y.append(y)
            all_logits.append(logits)
        y = torch.cat(all_y, dim=0)
        logits = torch.cat(all_logits, dim=0)
        return (y == torch.argmax(logits, dim=1)).float().mean().item() * 100
        
