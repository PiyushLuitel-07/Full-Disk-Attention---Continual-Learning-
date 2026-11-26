import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import LinearAttentionBlock, ProjectorBlock
from initialize import *

"""
Architecture of the attention-based full-disk model. 
Passing attention=False will switch the model to standard CNN pipeline
"""
class Attn_Net(nn.Module):
    def __init__(self, im_size, num_classes, attention=True, normalize_attn=True, init='kaimingUniform'):
        super(Attn_Net, self).__init__()
        self.attention = attention
        
        # conv blocks
        self.conv_block1 = nn.Sequential(
                            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, bias=True),
                            nn.BatchNorm2d(num_features=64, affine=True),
                            nn.ReLU())
        self.conv_block2 = nn.Sequential(
                            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=True),
                            nn.BatchNorm2d(num_features=128, affine=True),
                            nn.ReLU())
        self.conv_block3 = nn.Sequential(
                            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=True),
                            nn.BatchNorm2d(num_features=256, affine=True),
                            nn.ReLU())
        self.conv_block4 = nn.Sequential(
                            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=True),
                            nn.BatchNorm2d(num_features=512, affine=True),
                            nn.ReLU())
        self.conv_block5 = nn.Sequential(
                            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True),
                            nn.BatchNorm2d(num_features=512, affine=True),
                            nn.ReLU())
        self.conv_block6 = nn.Sequential(
                            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=3, padding=0, bias=True),
                            nn.BatchNorm2d(num_features=512, affine=True),
                            nn.ReLU())
        
        self.dense = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, stride=2, padding=0, bias=True)
        
        # Projectors & Compatibility functions
        if self.attention:
            self.projector1 = ProjectorBlock(256, 512)
            self.attn1 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
            self.attn2 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
            self.attn3 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
            
        # final classification layer
        if self.attention:
            self.classify = nn.Linear(in_features=512*3, out_features=num_classes, bias=True)
            
        else:
            self.classify = nn.Linear(in_features=512, out_features=num_classes, bias=True)
            
        # initialize
        if init == 'kaimingNormal':
            weights_init_kaimingNormal(self)
        elif init == 'kaimingUniform':
            weights_init_kaimingUniform(self)
        else:
            raise NotImplementedError("Invalid type of initialization!")

    def forward(self, x):
        # feed forward
        x = self.conv_block1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0) # /2
        x = self.conv_block2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0) # /4
        l1 = self.conv_block3(x) # /1
        x = F.max_pool2d(l1, kernel_size=2, stride=2, padding=0) # /8
        l2 = self.conv_block4(x) # /2
        x = F.max_pool2d(l2, kernel_size=2, stride=2, padding=0) # /16
        l3 = self.conv_block5(x) # /4
        x = F.max_pool2d(l3, kernel_size=2, stride=2, padding=0) # /32
        x = self.conv_block6(x)
        g = self.dense(x) # batch_sizex512x1x1
#         print(g.shape)
        # pay attention
        if self.attention:
            c1, g1 = self.attn1(self.projector1(l1), g)
            c2, g2 = self.attn2(l2, g)
            c3, g3 = self.attn3(l3, g)
            g = torch.cat((g1,g2,g3), dim=1) # batch_sizexC
            # classification layer
            x = self.classify(g) # batch_sizexnum_classes
        else:
            c1, c2, c3 = None, None, None
            x = self.classify(torch.squeeze(g))
        return [x, c1, c2, c3]
