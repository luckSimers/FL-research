import logging
import time
from torch.utils.data import DataLoader
from utils import *
from sklearn.metrics import confusion_matrix
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_

class Centralized(object):

    def __init__(self, args) -> None:
        self.algorithm = args.algorithm
        self.best_acc = 0
        self.current_round = 0
        self.dataset = args.dataset
        self.device = torch.device('cuda:0')
        self.exp_tag = args.exp_tag
        self.global_rounds = args.global_rounds
        self.logger = args.logger
        self.load_path = args.load_path
        self.net = args.net
        self.save_dir = args.save_dir
        self.make_dataset(args)
        args.class_num = self.class_num
        self.train_loader = DataLoader(self.trainset, batch_size=args.bs, shuffle=True)
        self.test_loader = DataLoader(self.testset, batch_size=args.eval_bs, shuffle=False)
        self.make_model()
        self.optimizer = get_optimizer(self.model, optim_name=args.optim, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_training_steps=self.global_rounds, num_warmup_steps=0)
        self.clip_grad = args.clip_grad
        self.loss = nn.CrossEntropyLoss()

    def train(self, round_idx):
        st = time.time()
        self.model.train(True)
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            if len(y) < 2:
                continue
            logits = self.model(x)
            loss = self.loss(logits, y)
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optimizer.step()
        logging.info(f'>> training cost {(time.time() - st)/60:.2f} min')

    def make_model(self):
        logging.debug('Building models')
        net_builder = get_net_builder(self.net)
        self.model = net_builder(self.class_num).to(self.device)
        
    def make_dataset(self, args):
        logging.debug('Loading and spliting dataset')
        data_info = fetch_fl_dataset(args.dataset, args.data_dir, client_num=1)
        self.trainset, self.testset = data_info['train_ds'], data_info['test_ds']
        self.class_num = data_info['class_num']

        if args.compress_freq:
            self.trainset = Subset4FL(
                self.dataset, self.trainset, None,
                self.testset.transform,
                compress_freq=args.compress_freq,
                randomize=args.randomize,
                del_num=args.del_freq_chs
            )
        
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
            logging.info(f'Round {round_idx}/{self.global_rounds}...')
            st_time = time.time()
            self.train(round_idx)
            self.scheduler.step()
            self.test(round_idx)
            logging.info(f'time cost {(time.time() - st_time)/60:.2f} min')
    

    @torch.no_grad()
    def test(self, round_idx):
        logging.info('Testing at round {}/{}'.format(round_idx, self.global_rounds))
        st = time.time()
        loader = self.test_loader
        all_y, all_logits = [], []
        self.model.train(False)
        self.model.to(self.device)
        for x, y in loader:   
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
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
        if best:
            self.save_model(round_idx, best=best)
        logging.info(f"server_test_time>> {(time.time() - st) / 60:.2f} min")


    def save_model(self, round, best=False):
        save_path = os.path.join(self.save_dir, self.exp_tag)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if best:
            path = os.path.join(save_path, f'{self.net}_models_best.pth')
            ckpt = {
                'ckpt_model': self.model.state_dict(),
            }
            torch.save(ckpt, path)
        else:
            path = os.path.join(save_path, f'{self.net}_{round}.pth')
            ckpt = {
                'ckpt_model': self.model.state_dict(),
            }
            torch.save(ckpt, path) 


    @staticmethod
    def get_argument():
        return [
        ]