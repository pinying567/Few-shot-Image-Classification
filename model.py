import torch
from torch import nn

class Convnet(nn.Module):
    def __init__(self, in_channels=3, hid_channels=64, out_channels=64):
        super(Convnet, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, hid_channels)
        )
    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)
    
def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class MLP(nn.Module):
    def __init__(self, in_dim=1600, hidden_dim=1024):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(hidden_dim//2, 1)
        )
        
    def forward(self, x1, x2, dim=0):
        x = torch.cat((x1, x2), dim=dim)
        x = self.mlp(x).squeeze()
        return x

class Hallucinator(nn.Module):
    def __init__(self, dim=1600):
        super(Hallucinator, self).__init__()
        self.norm = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU()
        )
        # initialize as identity matrix
        nn.init.eye_(self.norm[0].weight)
        nn.init.zeros_(self.norm[0].bias)
        nn.init.eye_(self.norm[2].weight)
        nn.init.zeros_(self.norm[2].bias)
        
    def forward(self, x, n):
        x = self.norm(x)        
        x = torch.cat((x, n), dim=2)
        x = self.out(x)
        return x

# wGAN
class w_Discriminator(nn.Module):
    '''
    input (B, 1600)
    output (B, )
    '''
    def __init__(self, in_dim=1600, hidden_dim=1024):
        super(w_Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim//2, 1),
        )
        
    def forward(self, x):
        x = self.dis(x)
        return x

"""
import pdb
x1 = torch.randn([50, 5, 1600]).cuda()
x2 = torch.randn([50, 5, 1600]).cuda()
model = MLP().cuda()
out = model(x1, x2, dim=2)
pdb.set_trace()
"""
"""
import pdb
x = torch.randn([5, 1600]).cuda()
model = Hallucinator().cuda()
noise = torch.randn([5, 1600]).cuda()
out = model(x, noise)
pdb.set_trace()
"""