import torch
from torch import nn
import torch.nn.functional as F

class OTLayer(nn.Module):
    def __init__(self, lambd=1):
        super(OTLayer,self).__init__()
        self.lambd = lambd

    def one_hot(self, x):
        max_idx = torch.argmax(x, -1, keepdim=True)
        one_hot = torch.zeros_like(x)
        one_hot = one_hot.scatter_(-1,max_idx,1.).to(x.device)
        return one_hot

    def forward(self, x, y, L=100):
        B, C = x.shape
        x = x.reshape(B, -1)
        y = y.reshape(B, -1)
        cost = (x**2).sum(-1, keepdims=True) -2*F.linear(x, y)+(y**2).sum(-1)
        K = -cost*self.lambd
        log_a = torch.zeros(B).to(x.device)
        log_b = torch.zeros(B).to(x.device)
        log_u = torch.zeros_like(log_a)
        log_v = torch.zeros_like(log_b)
        l = 0
        while l < L:
            log_u = log_a - torch.logsumexp(K+log_v[None,:], dim=1)
            log_v = log_b - torch.logsumexp(K+log_u[:,None], dim=0)
            l += 1
        P = torch.exp(log_u.reshape(-1, 1) + K + log_v.reshape(1, -1))
        P = self.one_hot(P)
        return F.linear(P, y.T)

if __name__ == "__main__":
    ot = OTLayer(1)
    x = torch.randn(100,2)*3
    y = torch.randn(100,2)
    y = ot(x, y, 100)
    import matplotlib.pyplot as plt
    for i in range(100):
        t1 = [x[i,0], y[i,0]]
        t2 = [x[i,1], y[i,1]]
        plt.plot(t1, t2)
    plt.show()
