import torch
import numpy as np
class Priorbuffer():
    def __init__(self, memory, prior):
        self.memory = memory
        self.buf_s = torch.zeros((self.memory, 3, 3)).cuda()
        self.buf_a = torch.zeros((self.memory, 3, 3)).long().cuda()
        self.buf_r = torch.zeros(self.memory).cuda()
        self.buf_s_ = torch.zeros((self.memory, 3, 3)).cuda()
        self._n = 0
        self._p = 0
        self.prior = prior

    def append(self, s, a, r, s_):
        self.buf_s[self._p, :, :] = s
        self.buf_a[self._p, :, :] = a
        self.buf_r[self._p] = r
        self.buf_s_[self._p, :, :] = s_
        self._p = (self._p + 1) % self.memory
        self._n = min(self._n + 1, self.memory)

    def order_accord_r(self):
        self.buf_r, indices = torch.sort(self.buf_r, descending=True)
        self.buf_s = self.buf_s[indices]
        self.buf_a = self.buf_a[indices]
        self.buf_s_ = self.buf_s_[indices]
        # print(self.buf_r.size())
        # print(self.buf_s.size())
        # print(self.buf_a.size())
        # print(self.buf_s_.size())

    def sample(self, batch_size):
        self.order_accord_r()
        indices = np.random.choice(self.prior, batch_size, replace=False)
        # print("indices=", indices)
        return self.buf_s[indices, :, :], self.buf_a[indices, :, :], self.buf_r[indices], self.buf_s_[indices, :, :]

    def reset(self):
        self.buf_s = torch.zeros((self.memory, 3, 3)).cuda()
        self.buf_a = torch.zeros((self.memory, 3, 3)).long().cuda()
        self.buf_r = torch.zeros(self.memory).cuda()
        self.buf_s_ = torch.zeros((self.memory, 3, 3)).cuda()
        self._n = 0
        self._p = 0


