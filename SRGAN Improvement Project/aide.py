import os
import torch
from Network import arch

def weights(pathtowgh):
    ntw = torch.load(pathtowgh)
    ncl = {}
    for x, y in ntw.items():
        if x.startswith('module.'):
            ncl[x[7:]] = y
        else:
            ncl[x] = y
    return ncl

def transfer_weights(trc, nte):
    tbd = [a for a in trc.keys()]

    for a, y in trc.items():
        if a in nte and nte[a].size() == y.size():
            trc[a] = nte[a]
            tbd.remove(a)

    trc['conv_first.weight'] = nte['model.0.weight']
    trc['conv_first.bias'] = nte['model.0.bias']

    for a in tbd.copy():
        if 'RDB' in a:
            ird = a.replace('RRDB_trunk.', 'model.1.sub.')
            if '.weight' in a:
                ird = ird.replace('.weight', '.0.weight')
            elif '.bias' in a:
                ird = ird.replace('.bias', '.0.bias')
            trc[a] = nte[ird]
            tbd.remove(a)

    slt = {
        'trunk_conv': 'model.1.sub.23',
        'upconv1': 'model.3',
        'upconv2': 'model.6',
        'HRconv': 'model.8',
        'conv_last': 'model.10',
    }

    for mlt, wce in slt.items():
        trc[f'{mlt}.weight'] = nte[f'{wce}.weight']
        trc[f'{mlt}.bias'] = nte[f'{wce}.bias']

    return trc

def main():
    mdlt = './models/RRDB_ESRGAN_x4.pth'
    sdtr = './models/RRDB_ESRGAN_x4.pth'

    ttlx = arch(3, 3, 64, 23, ci=32)
    mlbc = ttlx.state_dict()

    oix = weights(mdlt)

    mlbc = transfer_weights(mlbc, oix)

    torch.save(mlbc, sdtr)
    print('storing in ', sdtr)

if __name__ == "__main__":
    main()
