import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'ConvLayer',
    'PrimaryCaps',
    'DigitCaps',
    'Decoder',
]


# ====================================================================================================
# ====================================================================================================

class ConvLayer(nn.Module):
    '''A Convolutional Layer'''
    
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9, stride=1):
        super(ConvLayer, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        
    def forward(self, x):
        return F.relu(self.conv(x))

# ====================================================================================================

class PrimaryCaps(nn.Module):
    '''Primary Capsules'''
    
    def __init__(self, num_capsules=32, in_channels=256, out_channels=8, kernel_size=9, stride=2):
        super(PrimaryCaps, self).__init__()
        
        self.num_capsules = num_capsules
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, stride) for _ in range(num_capsules)
        ])
    
    def forward(self, x):
        output = []
        
        for capsule in self.capsules:
            out = F.relu(capsule(x))
            output.append(out)
        
        output = torch.stack(output, dim=1)
        output = output.view(x.size(0), -1, self.out_channels)
        
        return self.squash(output)
    
    def squash(self, x):
        '''Squash'''
        
        squared_norm = (x**2).sum(-1, keepdim=True) + 1e-18
        
        scale = squared_norm / (1 + squared_norm)
        unit = x / torch.sqrt(squared_norm)
        
        return scale * unit

# ====================================================================================================

class DigitCaps(nn.Module):
    '''Digit Capsules'''
    
    def __init__(self, num_capsules=10, num_inputs_per_capsule=32*6*6, in_dim=8, out_dim=16, r=3, cuda=False, sigma=0.01):
        super(DigitCaps, self).__init__()
        
        self.num_capsules = num_capsules
        self.num_inputs_per_capsule = num_inputs_per_capsule
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.r = r
        self.cuda = cuda
        self.sigma = sigma
        
        self.W = nn.Parameter(
            sigma * torch.randn(num_inputs_per_capsule, num_capsules, out_dim, in_dim)
        )
    
    def routing(self, u_hat):
        '''Dynamic Routing (Routing By Agreement)'''
        
        b_ij = torch.zeros(u_hat.size(0), self.num_inputs_per_capsule, self.num_capsules)
        if self.cuda and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        b_ij = b_ij.to(device)
        
        for iteration in range(self.r):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = c_ij.unsqueeze(3).unsqueeze(4)
            
            if iteration == self.r - 1:
                s_j = (u_hat @ c_ij).sum(dim=1)
                v_j = self.squash(s_j)
            else:
                with torch.no_grad():
                    s_j = (u_hat @ c_ij).sum(dim=1)
                    v_j = self.squash(s_j)
                    
                    d = u_hat.transpose(3, 4) @ torch.stack([v_j] * self.num_inputs_per_capsule, dim=1)
                    b_ij = b_ij + d.squeeze(4).squeeze(3).mean(dim=0)
        
        return v_j.squeeze(3)
        
    def forward(self, x):

        # Get the dimensions right
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)
        W = torch.stack([self.W] * x.size(0), dim=0)
        
        # Affine Transformation
        u_hat = W @ x 
        
        # Dynamic Routing
        return self.routing(u_hat)
        
    def squash(self, x):
        '''Squash'''
        
        squared_norm = (x**2).sum(-1, keepdim=True) + 1e-18
        
        scale = squared_norm / (1 + squared_norm)
        unit = x / torch.sqrt(squared_norm)
        
        return scale * unit

# ====================================================================================================

class Decoder(nn.Module):
    '''Capsule Net Decoder'''
    
    def __init__(self, num_capsules=10, in_dim=16, out_dim=28*28):
        super(Decoder, self).__init__()
        
        self.num_capsules = num_capsules
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.reconstruction_layers = nn.Sequential(
            nn.Linear(num_capsules*in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x, y=None):
        if y is None:
            y = x.norm(dim=-1).argmax(dim=-1)
        mask = F.one_hot(y, self.num_capsules).float().unsqueeze(-1)
        
        return self.reconstruction_layers((x * mask).view(x.size(0), -1))

# ====================================================================================================
