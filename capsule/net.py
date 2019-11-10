import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvLayer, Decoder, DigitCaps, PrimaryCaps

__all__ = [
    'CapsNet',
]


class CapsNet(nn.Module):
    '''Capsule Network'''
    
    def __init__(self, num_capsules=10, in_channels=1, num_inputs_per_capsule=32*6*6, out_dim=28*28, cuda=False):
        super(CapsNet, self).__init__()

        CapsNet.num_capsules = num_capsules
        
        self.conv_layer = ConvLayer(in_channels=in_channels)
        self.primary_capsules = PrimaryCaps()
        self.digit_capsules = DigitCaps(num_capsules=num_capsules, num_inputs_per_capsule=num_inputs_per_capsule, cuda=cuda)
        self.decoder = Decoder(num_capsules=num_capsules, out_dim=out_dim)

    def forward(self, x, y=None):
        x = self.conv_layer(x)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x)
        
        reconstruction = self.decoder(x, y)
        
        return x, x.norm(dim=-1), reconstruction
    
    @staticmethod
    def loss(output, targets, reconstruction, images):
        return CapsNet.margin_loss(output, targets) + \
            CapsNet.reconstruction_loss(reconstruction, images)

    @staticmethod
    def margin_loss(output, targets, lmbd=0.5, m_plus=0.9, m_minus=0.1):
        targets = F.one_hot(targets, CapsNet.num_capsules).float()
        
        left = torch.clamp(m_plus - output, min=0)**2
        right = torch.clamp(output - m_minus, min=0)**2
        
        loss = targets * left + lmbd*(1 - targets) * right
        return loss.sum(dim=1).mean()
    
    @staticmethod
    def reconstruction_loss(reconstruction, images, criterion=nn.MSELoss(), alpha=0.0005):
        return alpha*criterion(reconstruction, images.view(reconstruction.size(0), -1))
