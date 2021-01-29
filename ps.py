from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import ray
import torch
import torch.nn as nn
import torchvision
import torchvision.models as torchmodels
from filelock import FileLock
from torchvision import transforms
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group
)
import logging
import socket
logger = logging.Logger(__name__)
logger.setLevel("DEBUG")


@ray.remote(num_gpus=1, num_cpus=1)
class Worker(object):
    def __init__(self,
                 model='vgg16',
                 batch_size=128):
        # torch.manual_seed(0)
        # np.random.seed(0)
        model='vgg16'
        self.model_type = model
        print("=> creating model '{}'".format(model))
        self.model = torchmodels.__dict__[model]().cuda()
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.batch_size = batch_size
        self.train_loader = self.get_data_loader(self.batch_size)
        self.optimizer = None
        self.params = None
        return socket.gethostname()

    def num_params(self):
        return len(self.get_weights())

    def params_distribution(self):
        logger.info("getting")
        distribution = []
        weights = self.get_weights()
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

    @ray.method(num_returns=2)
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
        params = self.get_weights()
        num_shards = np.unique(np.array(assignments)).size
        logger.info(num_shards)
        shards = [dict() for i in range(num_shards)]
        for i, (k, v) in enumerate(params.items()):
            shards[assignments[i]][k] = v.data.cpu()  # this will only be used by ps which locates on cpus
            # shards[assignments[i]][k] = v  # this will only be used by ps which locates on cpus
        return shards

    def index_shard(self, shards, index):
        return shards[index]

    def stitch_parameters(self, *split_params):
        # need to construct a weight dict
        params = dict()
        for p in split_params:
            for k, v in p.items():
                params[k] = v
        return params

    def set_weights(self, weights):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.copy_(weights[name])
        return True

    def get_weights(self):
        param_dict = {}
        for name, param in self.model.named_parameters():
            param_dict[name] = param
        return param_dict

    def get_gradients(self):
        grad_dict = {}
        for name, p in self.model.named_parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            # grad = None if p.grad is None else p.grad
            grad_dict[name] = grad
        return grad_dict

    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = params
        self.optimizer = torch.optim.SGD(self.params.values(), lr=0.001)
        return True

    def apply_updates(self, *list_of_gradients):
        assert(len(list_of_gradients) >= 1)
        summed_gradient_dict = dict()
        for name in self.params:
            summed_gradient_dict[name] = \
                np.stack([grads[name] for grads in list_of_gradients]).sum(axis=0)
        self.optimizer.zero_grad()
        self._set_gradients(summed_gradient_dict)
        self.optimizer.step()
        return True

    def _set_gradients(self, gradients):
        # gradients should be a stitched dict
        for name, p in self.get_params().items():
            if gradients[name] is not None:
                if p.grad is not None:
                    p.grad = torch.from_numpy(gradients[name]).to(p.grad.device)
                else:
                    p.grad = torch.from_numpy(gradients[name])

@ray.remote(num_cpus=1)
class PS(object):
    def __init__(self):
        # torch.manual_seed(0)
        # np.random.seed(0)
        self.params = None
        self.optimizer = None

    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = params
        self.optimizer = torch.optim.SGD(self.params.values(), lr=0.001)
        return True

    def apply_updates(self, *list_of_gradients):
        assert(len(list_of_gradients) >= 1)
        summed_gradient_dict = dict()
        for name in self.params:
            summed_gradient_dict[name] = \
                np.stack([grads[name] for grads in list_of_gradients]).sum(axis=0)
        self.optimizer.zero_grad()
        self._set_gradients(summed_gradient_dict)
        self.optimizer.step()
        return True

    def _set_gradients(self, gradients):
        # gradients should be a stitched dict
        for name, p in self.get_params().items():
            if gradients[name] is not None:
                if p.grad is not None:
                    p.grad = torch.from_numpy(gradients[name]).to(p.grad.device)
                else:
                    p.grad = torch.from_numpy(gradients[name])


class PSStrategy(object):
    def __init__(self,
                 num_worker=1,
                 num_ps=1,
                 model='resnet50',
                 batch_size=128):
        self.num_ps = num_ps
        self.num_worker = num_worker
        self.model = model
        #gpu_bundle = {"GPU": num_worker}
        #extra_resource_bundle = {"CPU": num_ps}
        #pg = placement_group([gpu_bundle, extra_resource_bundle], strategy="STRICT_PACK")
        #ray.get(pg.ready())
        #self.workers = [Worker.options(
        #placement_group=pg,
        #placement_group_bundle_index=0) # Index of gpu_bundle is 0.
        #.remote(model=self.model, batch_size = batch_size) for _ in range(self.num_worker)]
        #self.servers = [PS.options(
        #placement_group=pg,
        #placement_group_bundle_index=1) # Index of extra_resource_bundle is 1.
        #.remote() for _ in range(self.num_ps)]
        self.workers = [Worker.remote(model=self.model, batch_size=batch_size)
                        for i in range(self.num_worker)]
        self.servers = [PS.remote() for i in range(self.num_ps)]
        self.assignments = None
    
        self.initialize()

    def _round_robin_sharding(self): 
        """Generate the assignment of variable to servers."""
        parameter_distribution = ray.get(self.workers[0].params_distribution.remote()) 
        # formula given by https://www.usenix.org/conference/osdi20/presentation/jiang
        M = sum(parameter_distribution)
        n = self.num_worker
        k = self.num_ps
        SS_worker = (2 * (n - 1) / (n ** 2 + k * n - 2 * k)) * M
        SS_server = ((n - k) / (n** 2 + k * n - 2 * k)) * M
        print(SS_worker, SS_server)
        assignments = [0 for _ in parameter_distribution] 
        loads = [0 for _ in range(n+k)]
        full = [False for _ in range(n+k)]
        for i, var_size in enumerate(parameter_distribution):
            ps_index = -1
            while ps_index == -1:
                random_index = np.random.choice(range(n + k))
                if not full[random_index]:
                    ps_index = random_index
            #min_ps_index = loads.index(min(loads)) 
            loads[ps_index] += var_size
            if ps_index in range(n): # it is a worker PS
                if loads[ps_index] > SS_worker:
                    full[ps_index] = True
            else:
                if loads[ps_index] > SS_server:
                    full[ps_index] = True

            assignments[i] = ps_index 
        print("Load of each ps{}".format(loads)) 
        self.assignments = assignments

    def initialize(self):
        # All sync with worker 0
        init_weights_id = self.workers[0].get_weights.remote()
        
        self._round_robin_sharding()
        # all workers get synced
        for i, worker in enumerate(self.workers): 
            if i != 0:
                ray.wait([worker.set_weights.remote(init_weights_id)])
        # now spawn parameter server actors
        shard_ids = self.workers[0].split_parameters.remote(self.assignments)
        for i, worker_server in enumerate(self.workers):
            this_shard_id = self.workers[0].index_shard.remote(shard_ids, i)
            ray.wait([worker_server.set_params.remote(this_shard_id)])

        for i, server in enumerate(self.servers):
            this_shard_id = self.workers[0].index_shard.remote(shard_ids, i+self.num_worker)
            ray.wait([server.set_params.remote(this_shard_id)])

    def step(self):
        # stitch parameters
        param_ids = [w.get_params.remote() for w in self.workers]
        param_ids_server = [ps.get_params.remote() for ps in self.servers]
        param_ids.extend(param_ids_server)
        # worker compute the grads
        ps_grad_mappings = [list() for i in range(self.num_worker + self.num_ps)]
        loss_vals = []
        for i in range(5): 
            for worker in self.workers:
                stitched_param_id = worker.stitch_parameters.remote(*param_ids)
                grad_id, loss = worker.compute_gradients.remote(stitched_param_id)
                loss_vals.append(loss)
                split_gradient_ids = worker.split_gradients.remote(grad_id, self.assignments)
                for i in range(self.num_worker):
                    this_shard_id = worker.index_shard.remote(split_gradient_ids, i)
                    ps_grad_mappings[i].append(this_shard_id)
                for i in range(self.num_ps):
                    this_shard_id = worker.index_shard.remote(split_gradient_ids, i+self.num_worker)
                    ps_grad_mappings[i+self.num_worker].append(this_shard_id)
        ret = [w.apply_updates.remote(*ps_grad_mappings[i]) for i, w in enumerate(self.workers)]
        ret_server = [ps.apply_updates.remote(*ps_grad_mappings[i+self.num_worker]) for i, ps in enumerate(self.servers)]
        ret.extend(ret_server)
        ray.wait(ret)
        return ray.get(loss_vals)
