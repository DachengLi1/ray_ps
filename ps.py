from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import time
import ray
import torch
import torch.nn as nn
import torchvision
import torchvision.models as torchmodels
from filelock import FileLock
from torchvision import transforms
from ray.util import collective
import logging

@ray.remote(num_gpus=1, num_cpus=1)
class Worker(object):
    def __init__(self,
                 model,
                 batch_size,
                 world_size,
                 rank,
                 num_ps):
        # torch.manual_seed(0)
        # np.random.seed(0)
        self.model_type = model
        print("=> creating model '{}'".format(model))
        self.model = torchmodels.__dict__[model]().cuda()
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.batch_size = batch_size
        self.train_loader = self.get_data_loader(self.batch_size)
        self.world_size = world_size
        self.rank = rank
        self.num_ps = num_ps
        self.num_workers = self.world_size - self.num_ps
        self.assignments = None
        # index i of this list stores the names of params in ith server.
        self.name_list = [[] for i in range(num_ps)]
        collective.init_collective_group(world_size, rank, "nccl", "default")

    def num_params(self):
        return len(self.get_weights())

    def params_distribution(self):
        distribution = []
        weights = self.get_weights(cpu=True)
        for k, v in weights.items():
            distribution.append(v.numel())
        return distribution

    def get_data_loader(self, batch_size):
        # We add FileLock here because multiple workers will want to
        # download data, and this may cause overwrites since
        # DataLoader is not threadsafe.
        with FileLock(os.path.expanduser("/tmp/data.lock")):
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            trainset = torchvision.datasets.CIFAR10(root='/tmp/', train=True,
                                                    download=True, transform=transform_train)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                      shuffle=True, num_workers=2)

        return trainloader

    def compute_gradients(self, weights):
        x, y = next(iter(self.train_loader))
        x = x.cuda()
        y = y.cuda()
        self.set_weights(weights)
        self.model.zero_grad()
        output = self.model(x)
        loss = self.criterion(output, y)
        loss.backward()
        return self.get_gradients(), loss.cpu().data.numpy()

    def split_gradients(self, grad, assignments):
        # assuming messages are gradients or parameters
        # this grad is ready to be called by apply_gradients in ParameterServer
        num_shards = np.unique(np.array(assignments)).size
        shards = [dict() for i in range(num_shards)]
        for i, (k, v) in enumerate(grad.items()):
            shards[assignments[i]][k] = v
        return shards

    def split_parameters(self, assignments):
        params = self.get_weights(cpu=False)
        num_shards = np.unique(np.array(assignments)).size
        shards = [dict() for i in range(num_shards)]
        for i, (k, v) in enumerate(params.items()):
            shards[assignments[i]][k] = v.data.cpu()  # this will only be used by ps which locates on cpus
            # shards[assignments[i]][k] = v  # this will only be used by ps which locates on cpus
        return shards

    def index_shard(self, shards, index):
        return shards[index]

    def set_weights(self, weights):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                weight = weights[name].cuda()
                param.copy_(weight)
        return True

    def get_weights(self, cpu):
        param_dict = {}
        for name, param in self.model.named_parameters():
            if cpu:
                param_dict[name] = param.data.cpu()
            else:
                param_dict[name] = param
        return param_dict

    def get_gradients(self):
        grad_dict = {}
        for name, p in self.model.named_parameters():
            # grad = None if p.grad is None else p.grad
            grad_dict[name] = p.grad
        return grad_dict

    def set_assignments(self, assignments):
        self.assignments = assignments
        keys = list(self.get_weights(cpu=False).keys())
        for i, a in enumerate(self.assignments):
            self.name_list[a].append(keys[i])

    def compute(self):
        """Returns the loss, and send gradients to servers"""
        # First receive params from servers
        param_shards = []
        weights = self.get_weights(cpu=False)
        params = dict()
        for i in range(self.num_ps):
            # the parameter set of this server
            param_shard_keys = self.name_list[i]
            for key in param_shard_keys:
                to_recv = weights[key]
                param_recv = torch.zeros(to_recv.size()).cuda()
                collective.recv(param_recv, self.num_workers+i, "default")
                params[key] = param_recv

        grad, loss = self.compute_gradients(params)

        # send this grad to every server
        split_grad = self.split_gradients(grad, self.assignments)
        for i in range(self.num_ps):
            this_shard = self.index_shard(split_grad, i)
            for _, v in this_shard.items():
                collective.send(v, self.num_workers+i, "default")

        return loss

@ray.remote(num_cpus=1, num_gpus=1)
class PS(object):
    def __init__(self, workers, world_size, rank):
        # torch.manual_seed(0)
        # np.random.seed(0)
        self.params = dict()
        self.optimizer = None
        self.workers = workers
        self.world_size = world_size
        self.rank = rank
        collective.init_collective_group(self.world_size, self.rank, "nccl", "default")

    def send_params(self, dst_rank):
        """ Send this param shard to the destination worker """
        for _, v in self.params.items():
            collective.send(v, dst_rank, "default")

    def get_params(self):
        return self.params

    def set_params(self, params):
        for k, v in params.items():
            self.params[k] = v.cuda()
        self.optimizer = torch.optim.SGD(self.params.values(), lr=0.001)
        return True

    def _set_gradients(self, gradients):
        # gradients should be a stitched dict
        for name, p in self.get_params().items():
            if gradients[name] is not None:
                p.grad = gradients[name]
    
    def update(self, src_rank):
        """Receive gradients and update"""
        keys = self.params.keys()
        grads = dict()
        for key in keys:
            to_recv = self.params[key]
            grad_recv = torch.zeros(to_recv.size()).cuda()
            collective.recv(grad_recv, src_rank, "default")
            grads[key] = grad_recv
        
        self.optimizer.zero_grad()
        self._set_gradients(grads)
        self.optimizer.step()
        return True

class PSStrategy(object):
    def __init__(self,
                 num_worker=1,
                 num_ps=1,
                 model='resnet50',
                 batch_size=128):
        self.num_ps = num_ps
        self.num_worker = num_worker
        self.model = model
        self.world_size = self.num_ps + self.num_worker
        self.workers = [Worker.remote(model=self.model, batch_size=batch_size, world_size=self.world_size, rank=i, num_ps=self.num_ps) 
                        for i in range(self.num_worker)]
        self.servers = [PS.remote(workers=self.workers, world_size=self.world_size, rank=i+self.num_worker) for i in range(self.num_ps)]
        self.assignments = None

        self.initialize()

    def _round_robin_sharding(self):
        """Generate the assignment of variable to servers."""
        parameter_distribution = ray.get(self.workers[0].params_distribution.remote())
        assignments = [0 for _ in parameter_distribution]
        loads = [0 for _ in range(self.num_ps)]
        for i, var_size in enumerate(parameter_distribution):
            min_ps_index = loads.index(min(loads))
            loads[min_ps_index] += var_size
            assignments[i] = min_ps_index
        print("Load of each ps {}".format(loads))
        self.assignments = assignments

    def initialize(self):
        # All sync with worker 0
        init_weights_id = self.workers[0].get_weights.remote(cpu=True)

        self._round_robin_sharding()

        # set assignments to every worker
        ray.wait([w.set_assignments.remote(self.assignments) for w in self.workers])

        # all workers get synced
        for i, worker in enumerate(self.workers):
            if i != 0:
                ray.wait([worker.set_weights.remote(init_weights_id)])

        # now spawn parameter server actors
        shard_ids = self.workers[0].split_parameters.remote(self.assignments)
        for i, server in enumerate(self.servers):
            this_shard_id = self.workers[0].index_shard.remote(shard_ids, i)
            ray.wait([server.set_params.remote(this_shard_id)])

    def step(self):
        loss_vals = []
        rets = []
        for worker in self.workers:
            for server in self.servers:
                # every server sends its shard to the worker
                server.send_params.remote(self.workers.index(worker))
            # the worker receives shards from ps, compute loss, gradients
            # and sends these gradients to every server
            loss = worker.compute.remote()
            for server in self.servers:
                rets.append(server.update.remote(self.workers.index(worker)))
            loss_vals.append(loss)
        ray.wait(rets)
        return ray.get(loss_vals)
