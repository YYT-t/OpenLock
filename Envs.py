import torch
import numpy as np
class BaseEnv():
    def __init__(self, order):
        '''
        -1: fail, 0: have not been explored, 1: success
        correct_mat has no 0, all -1 or 1
        '''
        self.order = order
        self.correct_mat = self.get_reasonable_mat()
        self.state = torch.zeros((self.order, self.order)).cuda()

    def step(self, a):
        '''
        :param a: order, order
        :return:
        '''
        rew = None
        s_by_a = torch.sum(self.state * a)
        cor_by_a = torch.sum(self.correct_mat * a)
        if s_by_a == 0 and cor_by_a == 1:
            # only correct and have not been discovered
            rew = 10
        elif s_by_a == 0 and cor_by_a == -1:
            rew = 0
        elif s_by_a != 0:
            rew = -10
        self.state = (1 - a) * self.state + a * self.correct_mat
        done = False
        if torch.sum((self.state == 0)) == 0 or torch.sum((self.state == 1)) == self.order-1:
            done = True
        return rew, self.state, done

    def get_reasonable_mat(self):
        res = -torch.ones((self.order, self.order)).cuda()
        strip = np.random.choice(range(self.order))
        eles = np.random.choice(range(self.order), size=self.order-1, replace=False)
        c_r = torch.rand(1)
        if c_r < 0.5:
            res[strip, eles] = 1
        else:
            res[eles, strip] = 1
        return res

    def reset(self):
        self.correct_mat = self.get_reasonable_mat()
        self.state = torch.zeros((self.order, self.order)).cuda()
        return self.state


