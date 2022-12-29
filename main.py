from Envs import BaseEnv
from Trainers import Base_trainer
import argparse
import torch
parser = argparse.ArgumentParser()
parser.add_argument('--buffer_memory', type=int, default=100)
parser.add_argument('--prior',  type=int, default=50)
parser.add_argument('--batch_size', type=int, default=40)
parser.add_argument('--train_epoch', type=int, default=1000)
parser.add_argument('--train_time_step', type=int, default=100)
args = parser.parse_args()

Env = BaseEnv(3)
Trainer = Base_trainer(env=Env, n_batch=args.batch_size, mem_cap=args.buffer_memory, prior=args.prior)
Epoch = args.train_epoch
T = args.train_time_step
for e in range(Epoch):
    s = Env.reset()
    R = 0
    for t in range(T):
        print("t=", t)
        print("s=", s)
        eps = torch.rand(1)
        if eps < 0.15:
            a_idx = torch.multinomial(torch.ones(3)/3, 1).squeeze()
            a = Trainer.decode_a_idx(a_idx)
        else:
            a = Trainer.get_a(s)
            # print("a=", a)
        r, s_, done = Env.step(a)
        # print("r=", r)
        R += r
        Trainer.buffer.append(s, a, r, s_)
        Trainer.update()
        if done:
            break
        s = s_
    print("R=", R)