from Networks import Convnet
import torch.optim as optim
from Buffers import Priorbuffer
import torch.nn.functional as F
import torch
class Base_trainer():
    def __init__(self, env, n_batch, mem_cap, prior):
        self.order = env.order
        self.action_num = self.order * self.order
        self.n_batch = n_batch
        self.mem_cap = mem_cap
        self.on_Q = Convnet().cuda()
        self.optimizer = optim.Adam(self.on_Q.parameters(), lr=0.01)
        self.buffer = Priorbuffer(self.mem_cap, prior)

    def update(self):
        self.on_Q.train()
        if self.buffer._n < self.n_batch:
            return
        batch_s, batch_a, batch_r, batch_s_ = self.buffer.sample(self.n_batch)
        # print("batch_r=", batch_r)
        y = batch_r  # b, 1
        batches = torch.zeros((batch_s.size(0), 2, self.order, self.order)).cuda()  # b, 2, 3, 3
        batches[:, 0, :, :] = batch_s
        batches[:, 1, :, :] = batch_a
        this_q = self.on_Q(batches)  # b
        loss = F.mse_loss(y, this_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print("loss=", loss)

    def decode_a_idx(self, a_idx):
        '''
        :param a_idx:
        :return: 3, 3
        '''
        res = torch.zeros((self.order, self.order)).cuda()
        row = int(a_idx / self.order)
        col = a_idx % self.order
        res[row, col] = 1
        return res

    def get_all_sa_pair(self, s):
        '''
        :param s: 3, 3
        :return: action_num, 2, order, order
        '''
        res = torch.zeros((self.action_num, 2, self.order, self.order)).cuda()
        for a_idx in range(self.action_num):
            res[a_idx, 0, :, :] = s
            res[a_idx, 1, :, :] = self.decode_a_idx(a_idx)
        return res

    def get_a(self, s):
        '''
        :param s: 3, 3
        :return:
        '''
        sa_pairs = self.get_all_sa_pair(s)   # 9, 2, 3, 3
        self.on_Q.eval()
        # print("sa_pairs.size()=", sa_pairs.size())
        q = self.on_Q(sa_pairs).squeeze()  # 9
        print("q=", q.view(3, 3))
        prob = F.softmax(q)
        # a_idx = torch.multinomial(prob, 1).squeeze()
        a_idx = torch.argmax(q)
        a = self.decode_a_idx(a_idx)
        return a