import torch
import torch.nn as nn
import torch.nn.functional as F

from capsulelayers import ConvLayer, Decoder, DigitCaps, PrimaryCaps

__all__ = [
    'CapsNet',
]


class CapsNet(nn.Module):
    '''Capsule Network'''
    
    num_capsules = 10
    
    def __init__(self, cuda=False):
        super(CapsNet, self).__init__()
        
        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCaps()
        self.digit_capsules = DigitCaps(cuda=cuda)
        self.decoder = Decoder()
        
        CapsNet.num_capsules = self.digit_capsules.num_capsules
        
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
