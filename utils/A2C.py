import yaml
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.multiprocessing as mp


from models.model import Model
from utils.Env import Env


with open('config/hyperparameter.yaml') as f:
    Config = yaml.safe_load(f)


gamma = Config.get("gamma")

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.model = Model()

    def pi(self, x, softmax_dim=1):
        x = F.relu(self.fc1(x))
        x, _ = self.model(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        _, v = self.model(x)
        return v

def worker(worker_id, master_end, worker_end):
    master_end.close()  # Forbid worker to use the master end for messaging
    Env = Env(Config)
    Env.seed(worker_id)

    while True:
        cmd, para = worker_end.recv()
        if cmd == 'step':
            ob, reward = Env.step(para)
            # if done:
            #     ob = Env.reset()
            worker_end.send((ob, reward))
        elif cmd == 'choose_action':
            action = Env.env.map_to_action(para)
            worker_end.send(action)
        elif cmd == 'reset':
            ob = Env.reset()
            worker_end.send(ob)
        elif cmd == 'reset_task':
            ob = Env.reset_task()
            worker_end.send(ob)
        elif cmd == 'close':
            worker_end.close()
            break
        elif cmd == 'get_spaces':
            worker_end.send((Env.observation_space, Env.action_space))
        else:
            raise NotImplementedError

class ParallelEnv:
    def __init__(self, n_train_processes):
        self.nenvs = n_train_processes
        self.waiting = False
        self.closed = False
        self.workers = list()
        master_ends, worker_ends = zip(*[mp.Pipe() for _ in range(self.nenvs)])
        self.master_ends, self.worker_ends = master_ends, worker_ends

        for worker_id, (master_end, worker_end) in enumerate(zip(master_ends, worker_ends)):
            p = mp.Process(target=worker,
                           args=(worker_id, master_end, worker_end))
            p.daemon = True
            p.start()
            self.workers.append(p)

        # Forbid master to use the worker end for messaging
        for worker_end in worker_ends:
            worker_end.close()

    def step_async(self, actions):
        for master_end, action in zip(self.master_ends, actions):
            master_end.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [master_end.recv() for master_end in self.master_ends]
        self.waiting = False
        obs, rews = zip(*results)
        return np.stack(obs), np.stack(rews)

    def reset(self):
        for master_end in self.master_ends:
            master_end.send(('reset', None))
        return np.stack([master_end.recv() for master_end in self.master_ends])

    def choose_action(self, prob):
        for master_end in self.master_ends:
            master_end.send(('choose_action', prob))
        return np.stack([master_end.recv() for master_end in self.master_ends])

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):  # For clean up resources
        if self.closed:
            return
        if self.waiting:
            [master_end.recv() for master_end in self.master_ends]
        for master_end in self.master_ends:
            master_end.send(('close', None))
        for worker in self.workers:
            worker.join()
            self.closed = True

def compute_target(v_final, r_lst):
    G = v_final.reshape(-1)
    td_target = list()

    for r in r_lst:
        G = r + gamma * G 
        td_target.append(G)

    return torch.tensor(td_target[::-1]).float()
