import torch
import torch.nn as nn
import torch.nn.functional as F
from u_net import Unet

class CNF(nn.Module):
    def __init__(self, image_size=32, channels=3, a_min=1e-3):
        super().__init__()
        self.a_min = a_min
        self.v_t = Unet(#ベクトル場の近似
            dim=image_size,
            dim_mults=(1,2,4,8),
            channels=channels,
            with_time_emb=True,
            resnet_block_groups=2,
        )
        self.s_t = Unet(#スコアの近似
            dim=image_size,
            dim_mults=(1,2,4,8),
            channels=channels,
            with_time_emb=True,
            resnet_block_groups=2,
        )

    def u_t(self, x, x_0, x_1, t):
        return (1-2*t)/t*(1-t)*(x-(t*x_1+(1-t)*x_0))+(x_1-x_0)
    
    def log_px_dt(self, x, x_0, x_1, t):
        return (t*x_1+(1-t)*x_0-x)/(t*(1-t))
    
    def forward(self, x_0, x_1, t):
        t = t.reshape(-1,1,1,1)
        x = t*x_1+(1-t)*x_0+t*(1-t)*torch.randn_like(x_0)
        target_v = self.u_t(x, x_0, x_1, t)
        target_s = self.log_px_dt(x, x_0, x_1, t)
        t = t.reshape(-1)
        pred_v = self.v_t(x, t)
        pred_s = self.s_t(x, t)
        return F.mse_loss(pred_v, target_v)+F.mse_loss(pred_s, target_s)

    def sample(self, x, step=100):
        h = 1/step
        time_steps = torch.linspace(0, 1, step+1)
        for t in time_steps:
            t = torch.tensor([t], device=x.device, dtype=torch.float)
            g_t = t*(1-t)
            u_t = self.v_t(x, t)+g_t**2/2*self.s_t(x, t)
            with torch.no_grad():
                x = x+h*u_t+h*g_t**2*torch.randn_like(x)
        return x