# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

from models.LinearClassifier import LinearClassifier
from save_representations import save_representations
from transforms import Transform, EvalTransform

from torch import nn, optim
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

from barlow_twins import BarlowTwins
from datasets import NominalCIFAR10ImageDataset
from visualize import draw_tsne_visualization

parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument('data', default='./', type=Path, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--workers', default=1, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.6, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.015, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='1024-1024-1024', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=20, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')


def main():
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    if 'SLURM_JOB_ID' in os.environ:
        # single-node and multi-node distributed training on SLURM cluster
        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        # find a common host name on all nodes
        # assume scontrol returns hosts in the same order on all nodes
        cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
        args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
        args.dist_url = f'tcp://{host_name}:58472'
    else:
        # single-node distributed training
        args.rank = 0
        args.dist_url = 'tcp://localhost:58472'
        args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def evaluate_by_linear_probing(loader, model, gpu):
    model.eval()
    X = np.zeros((0, 512))
    y = np.zeros(0)
    for images, target in loader:
        images = images.cuda(gpu, non_blocking=True)
        with torch.no_grad():
            X = np.append(X, model.module.backbone(images).cpu().numpy(), axis=0)
        y = np.append(y, target.numpy(), axis=0)
    clf = LogisticRegression(max_iter=2000)
    # clf = KNeighborsClassifier(n_neighbors=20)

    # normalize data
    X = X - X.mean(axis=0)
    X = X / (X.std(axis=0) + 1e-7)

    clf.fit(X, y)
    return clf.score(X, y)


    # linear_classifier = LinearClassifier(2048, 10).cuda(gpu)
    # linear_classifier.train()
    #
    # dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)
    #
    # loss_fn = nn.CrossEntropyLoss()
    #
    # optimizer = optim.Adam(linear_classifier.parameters(), lr=1e-4)
    #
    # for epoch in range(200):
    #     for x, y in tqdm(dataloader):
    #         x = x.cuda(gpu, non_blocking=True)
    #         y = y.cuda(gpu, non_blocking=True)
    #
    #         optimizer.zero_grad()
    #
    #         pred = linear_classifier(x)
    #         loss = loss_fn(pred, y)
    #         loss.backward()
    #         optimizer.step()
    #
    # correct = 0
    #
    # for x, y in dataloader:
    #     x = x.cuda(gpu, non_blocking=True)
    #     y = y.cuda(gpu, non_blocking=True)
    #     pred = linear_classifier(x)
    #     correct += (pred.argmax(dim=1) == y).sum().item()
    #
    # return correct / len(dataset)


def main_worker(gpu, args):
    args.rank += gpu
    torch.distributed.init_process_group(
        backend='gloo', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    model = BarlowTwins(args.projector, args.batch_size, args.lambd).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        start_epoch = 0

    draw_tsne_visualization(model, 0)
    # save_representations(model, 0)

    # # train_dataset = torchvision.datasets.ImageFolder(args.data / 'train', Transform())
    # # train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=Transform())
    # train_dataset = NominalCIFAR10ImageDataset(nominal_class=0, train=True, transform=Transform())
    # test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=EvalTransform())
    # sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # assert args.batch_size % args.world_size == 0
    # per_device_batch_size = args.batch_size // args.world_size
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=per_device_batch_size, num_workers=args.workers,
    #     pin_memory=True, sampler=sampler)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=per_device_batch_size)
    #
    # start_time = time.time()
    # scaler = torch.cuda.amp.GradScaler()
    # for epoch in range(start_epoch, args.epochs):
    #     model.train()
    #     sampler.set_epoch(epoch)
    #     # for step, (y1, y2) in enumerate(train_loader, start=epoch * len(train_loader)):
    #     #     y1 = y1.cuda(gpu, non_blocking=True)
    #     #     y2 = y2.cuda(gpu, non_blocking=True)
    #     #     adjust_learning_rate(args, optimizer, train_loader, step)
    #     #     optimizer.zero_grad()
    #     #     with torch.cuda.amp.autocast():
    #     #         loss = model.forward(y1, y2)
    #     #     scaler.scale(loss).backward()
    #     #     scaler.step(optimizer)
    #     #     scaler.update()
    #     #     if step % args.print_freq == 0:
    #     #         if args.rank == 0:
    #     #             stats = dict(epoch=epoch, step=step,
    #     #                          lr_weights=optimizer.param_groups[0]['lr'],
    #     #                          lr_biases=optimizer.param_groups[1]['lr'],
    #     #                          loss=loss.item(),
    #     #                          time=int(time.time() - start_time))
    #     #             print(json.dumps(stats))
    #     #             print(json.dumps(stats), file=stats_file)
    #
    #
    #
    #     print(f"Linear probing accuracy: {100 * evaluate_by_linear_probing(test_loader, model, gpu):.3f}%")
    #
    #     if args.rank == 0:
    #         # save checkpoint
    #         state = dict(epoch=epoch + 1, model=model.state_dict(),
    #                      optimizer=optimizer.state_dict())
    #         torch.save(state, args.checkpoint_dir / 'checkpoint.pth')
    # if args.rank == 0:
    #     # save final model
    #     torch.save(model.module.backbone.state_dict(),
    #                args.checkpoint_dir / 'resnet50.pth')


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])






if __name__ == '__main__':
    main()
