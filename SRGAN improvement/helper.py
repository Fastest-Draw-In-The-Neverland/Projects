import sys
import torch
from collections import OrderedDict

def idp(ssa, path2psnr='./models/RRDB_PSNR_x4.pth', model1='./models/RRDB_ESRGAN_x4.pth'):
    modarch2 = torch.load(path2psnr)
    modarch = torch.load(model1)
    
    nettr = OrderedDict()
    print(f'Interpolating with alpha = {ssa}')

    for x, iod in modarch2.items():
        dfx = modarch[x]
        nettr[x] = (1 - ssa) * iod + ssa * dfx

    ppi = f'./models/interp_{int(ssa*10):02d}.pth'
    torch.save(nettr, ppi)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <ssa>")
        sys.exit(1)

    qa = float(sys.argv[1])
    idp(qa)
