import torch
import torch.nn as nn
import pytorch_wavelets as pw

class DWTProcessor(nn.Module):
    def __init__(self, wave='sym6', levels=5, mode='symmetric'):
        super(DWTProcessor, self).__init__()
        self.wave = wave
        self.levels = levels
        self.mode = mode
        self.dwt = pw.DWT1D(wave=wave, J=levels, mode=mode)
        self.idwt = pw.IDWT1D(wave=wave, mode=mode)
    def forward(self, x,drop=True):
        yl, yh= self.dwt(x)
        if drop:
            partial_yh = torch.zeros_like(yh[0])
            yh[0]=partial_yh
        
        return yl, yh
    def inverse(self, yl, yh):
        reconstruct=self.idwt((yl, yh))
        return reconstruct
        # reconstruction_error = torch.norm(x - reconstruct)

        
