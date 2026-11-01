import torch
import torch.nn as nn
import torch.nn.functional as F
from mambo import *

class CTC(nn.Module):
    def __init__(self):
        super().__init__()
        self.ctconv = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 32),stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(8)
        )
        
        self.ctpool = nn.AvgPool2d(kernel_size=(1,4), stride=(1,2), padding=0)
    
    def forward(self, x):
        x = self.ctconv(x)
        x = self.ctpool(x)
        return x

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *config, max_norm=1, **kwconfig):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*config, **kwconfig)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)
# (S, H,C,T)->(S, H,T)    
class GCAP(nn.Module):
    def __init__(self, in_channels, N=22, depth_multiplier=2):
        super(GCAP, self).__init__()
        self.depthwise = Conv2dWithConstraint(in_channels, in_channels * depth_multiplier, kernel_size=(N, 1), stride=1, padding=0, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels * depth_multiplier, in_channels * depth_multiplier, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(in_channels * depth_multiplier)
        self.activation = nn.SELU()
    def forward(self, x):
        # print(f'GCAP input size: {x.size()}')
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
    
# (S, H,T)
class GTCC(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.avg1 = nn.Sequential(
            nn.AvgPool1d(kernel_size=4,stride=4, padding=0),
            nn.Dropout(p=0.25))

        self.tconv = nn.Sequential(
                nn.ZeroPad2d((3, 4, 0, 0)),
                nn.Conv1d(self.in_channels, self.in_channels * 4, kernel_size=8, stride=1, padding=0, groups=self.in_channels, bias=False),
                nn.Conv1d(self.in_channels * 4, self.in_channels * 2, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm1d(num_features=self.in_channels * 2),
                nn.SELU()
            )
        self.avg2 = nn.Sequential(
            nn.AvgPool1d(kernel_size=4,stride=4, padding=0),
            nn.Dropout(p=0.25))
    def forward(self, x):
        x = self.avg1(x)
        x = self.tconv(x)
        x = self.avg2(x)
        return x


class LinearWithConstraint(nn.Linear):
    def __init__(self, *config, max_norm=1, **kwconfig):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*config, **kwconfig)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(LinearWithConstraint, self).forward(x)

class classifier_n(nn.Module):
    def __init__(self,output_class):
        super().__init__()
        self.output_class = output_class
        self.classifier_1 = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(96*34, self.output_class, max_norm=0.25))
    def forward(self, x):
        return self.classifier_1(x)

class region(nn.Module):
    def __init__(self):
        super(region, self).__init__()
        self.center=torch.tensor([0,3,9,15,19,21])
        self.left=torch.tensor([1,2,6,7,8,13,14,18])
        self.right=torch.tensor([4,5,10,11,12,16,17,20])
        self.listx=[self.left,self.right,self.center]
        self.gcap_c = GCAP(8,N=len(self.center), depth_multiplier=2)
        self.gcap_l = GCAP(8,N=len(self.left), depth_multiplier=2)
        self.gcap_r = GCAP(8,N=len(self.right), depth_multiplier=2)
        self.gcap_multi = GCAP(16,N=3, depth_multiplier=2)
    def forward(self, x):
        x_left=x[:,:,self.listx[0],:]
        x_right=x[:,:,self.listx[1],:]
        x_center=x[:,:,self.listx[2],:]
        x_left=self.gcap_l(x_left)
        x_right=self.gcap_r(x_right)
        x_center=self.gcap_c(x_center)
        x=torch.cat([x_left,x_center,x_right],dim=2)
        x=self.gcap_multi(x)
        return x

class region2(nn.Module):
    def __init__(self):
        super(region2, self).__init__()
        self.center=torch.tensor([0,2,3,4,8,9,10,14,15,16,19])
        self.outer=torch.tensor([1,5,6,7,11,12,13,17,18,20,21])
        self.listx=[self.center,self.outer]
        self.gcap_c = GCAP(8,N=len(self.center), depth_multiplier=2)
        self.gcap_o = GCAP(8,N=len(self.outer), depth_multiplier=2)
        self.gcap_multi = GCAP(16,N=2, depth_multiplier=2)
    def forward(self, x):
        x_center=x[:,:,self.listx[0],:]
        x_outer=x[:,:,self.listx[1],:]
        # x_center=x[:,:,self.listx[2],:]
        x_center=self.gcap_c(x_center)
        x_outer=self.gcap_o(x_outer)
        # x_center=self.gcap_c(x_center)
        x=torch.cat([x_center,x_outer],dim=2)
        x=self.gcap_multi(x)
        return x




class beta_gate2(nn.Module):
    def __init__(self):
        super(beta_gate2, self).__init__()
        self.maxpool=nn.MaxPool1d(kernel_size=32,stride=16, padding=0)
        self.meanpool=nn.AvgPool1d(kernel_size=32,stride=16, padding=0)
        self.router=nn.Sequential(
            nn.Flatten(),
            nn.Linear(10*8, 2))
        self.sigmoid=nn.Sigmoid()
    def forward(self, beta):
        beta_max=self.maxpool(beta)
        beta_mean=self.meanpool(beta)
        beta_mm=torch.cat([beta_max,beta_mean],dim=1)
        # print(f'mu_mm size: {beta_mm.size()}')
        beta_mm=self.router(beta_mm)
        beta_gate=self.sigmoid(beta_mm)
        return beta_gate


class NeuroMambaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.MAMBO = MAMBO_Net()
        self.ctconv1 = CTC()
        self.ctconv2 = CTC()
        self.region = region()
        self.region2 = region2()
        self.gcap=GCAP(8,N=22, depth_multiplier=2)
        self.gtcc = GTCC(48)
        self.classifier = classifier_n(4)

        self.beta_gate=beta_gate2()
        
        
    def forward(self, x):
        x_re,wave_b= self.MAMBO(x.squeeze(dim=1))

        beta_wave=wave_b[:,1:6,:]
        beta_select=self.beta_gate(beta_wave)
        
        x1 = self.ctconv1(x_re.unsqueeze(dim=1))
        x2=self.ctconv2(x)

        x1 = self.gcap(x1)
        x_lr = self.region(x2)
        x_ft = self.region2(x2)
        x_region=torch.stack([x_lr,x_ft],dim=1)
        x_beta_s = torch.einsum('bi,bijkl->bjkl',beta_select,x_region)

        out = torch.cat([x1,x_beta_s],dim=1)
        out = self.gtcc(out.squeeze(dim=2))
        out = self.classifier(out)
        return out
    
