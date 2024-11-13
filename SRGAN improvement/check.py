import os.path as osp
import glob
import cv2
import numpy as np
import torch
from Network import network

def loading(path):
    pic = cv2.imread(path, cv2.IMREAD_COLOR)
    pic = pic * 1.0 / 255
    pic = torch.from_numpy(np.transpose(pic[:, :, [2, 1, 0]], (2, 0, 1))).float()
    return pic.unsqueeze(0)

def savedpic(op, lmb):
    op = np.transpose(op[[2, 1, 0], :, :], (1, 2, 0))
    op = (op * 255.0).round()
    cv2.imwrite('results/{:s}_rlt.png'.format(lmb), op)

def main():
    mod = 'models/RRDB_ESRGAN_x4.pth'  
    graphpu = torch.device('cuda')  
    lr = 'LR/*'
    mdl = network(3, 3, 64, 23, ci=32)
    mdl.load_state_dict(torch.load(mod), strict=1)
    mdl.eval()
    mdl = mdl.to(graphpu)

    print('Model path {:s}. \nTesting...'.format(mod))

    nd = 0
    for path in glob.glob(lr):
        nd += 1
        lmb = osp.splitext(osp.basename(path))[0]
        print(nd, lmb)

        lri = loading(path).to(graphpu)

        with torch.no_grad():
            op = mdl(lri).data.squeeze().float().cpu().clamp_(0, 1).numpy()

        savedpic(op, lmb)

if __name__ == "__main__":
    main()
