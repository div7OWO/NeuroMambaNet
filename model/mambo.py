import torch
import torch.nn as nn
import pytorch_wavelets as pw
import torch.nn.functional as F
from DWT import *
from mamba_ssm import Mamba2
def safe_downsample_dw(conv, x, pad=6):
        x = F.pad(x, (pad, pad), mode='reflect')
        return conv(x) 


class MAMBO_Net(nn.Module):

    def __init__(self, ):
        super(MAMBO_Net, self).__init__()
        self.waveProcessor = DWTProcessor(wave='sym6', levels=5, mode='symmetric')
        self.mambo_1=Mamba2(d_model=22, d_state=64, d_conv=4, expand=2, headdim=22, chunk_size=289)
        self.mambo_2=Mamba2(d_model=22, d_state=64, d_conv=4, expand=2, headdim=22, chunk_size=50)
        self.mambo_3=Mamba2(d_model=22, d_state=64, d_conv=4, expand=2, headdim=22, chunk_size=40)
        self.mambo_4=Mamba2(d_model=22, d_state=64, d_conv=4, expand=2, headdim=22, chunk_size=15)
        self.conv1 = nn.Sequential(nn.Conv1d(22,22,kernel_size=3,stride=2,padding=0,groups=22))
        self.conv2 = nn.Sequential(nn.Conv1d(22,22,kernel_size=3,stride=2,padding=0,groups=22))
        self.conv3 = nn.Sequential(nn.Conv1d(22,22,kernel_size=3,stride=2,padding=0,groups=22))
        self.dpconv1 = nn.Conv1d(22,22,kernel_size=1,stride=1,padding=0,groups=22)
        self.dpconv2 = nn.Conv1d(22,22,kernel_size=1,stride=1,padding=0,groups=22)
        self.dpconv3 = nn.Conv1d(22,22,kernel_size=1,stride=1,padding=0,groups=22)
        self.dpconv4 = nn.Conv1d(22,22,kernel_size=1,stride=1,padding=0,groups=22)
        self.dpconv_yl = nn.Conv1d(22,22,kernel_size=1,stride=1,padding=0,groups=22)


        # self.bn1=nn.BatchNorm1d(22)
    def init_weight(self):
         nn.init.kaiming_normal_(self.conv1[0].weight)
         nn.init.kaiming_normal_(self.conv2[0].weight)
         nn.init.kaiming_normal_(self.conv3[0].weight)
         nn.init.kaiming_normal_(self.dpconv1.weight)
         nn.init.kaiming_normal_(self.dpconv2.weight)
         nn.init.kaiming_normal_(self.dpconv3.weight)
         nn.init.kaiming_normal_(self.dpconv4.weight)
    def weight_constraint(self, parameters, min_value, max_value):
        for param in parameters:
            param.data.clamp_(min_value, max_value)     
 
    def forward(self, x):
        yl,yh=self.waveProcessor(x,drop=True)
        # print(yh[0].shape)
        y1=self.dpconv1(yh[1])
        y2=self.dpconv2(yh[2])
        y3=self.dpconv3(yh[3])
        y4=self.dpconv4(yh[4])
        b1=safe_downsample_dw(self.conv1, y1)
        # b1=self.conv1(y1)
        y2=y2+b1
        b2=safe_downsample_dw(self.conv2, y2)
        # b2=self.conv2(y2)
        y3=y3+b2
        b3=safe_downsample_dw(self.conv3, y3)
        # b3=self.conv3(y3)
        y4=y4+b3
        # weight=self.gate(y4+self.dpconv_yl(yl))
        y1_m= self.mambo_1(y1.permute(0,2,1))
        y2_m= self.mambo_2(y2.permute(0,2,1))
        y3_m= self.mambo_3(y3.permute(0,2,1))
        y4_m= self.mambo_4(y4.permute(0,2,1))
        y_re=[yh[0],y1_m.permute(0,2,1),y2_m.permute(0,2,1),y3_m.permute(0,2,1),y4_m.permute(0,2,1)]
        re_wave=self.waveProcessor.inverse(yl,y_re)
        return re_wave,y2_m.permute(0,2,1)