import torch
import numpy as np
import argparse
import os
import model as models
import torch.backends.cudnn as cudnn

from datasets import Get_Dataset
from utils import init_device_seed


def main():
    parser = argparse.ArgumentParser(description='Pedestrian Attribute Framework')
    parser.add_argument('--resume', default='', type=str, required=False, help='(default=%(default)s)')

    device = init_device_seed(1234, '0')

    args = parser.parse_args()
    # Data loading code
    train_dataset, val_dataset, test_dataset, attr_num, description = Get_Dataset("jagalchi", "inception_iccv")

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    model = models.__dict__["inception_iccv"](pretrained=True, num_classes=attr_num)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    print('')

    # model = model.to(device)
    model = torch.nn.DataParallel(model).to(device)

    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_accu = checkpoint['best_accu']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        return

    cudnn.benchmark = False
    cudnn.deterministic = True

    model.eval()

    tol = 0
    scores = np.empty((len(val_loader) * 32, attr_num))
    targets = np.empty((len(val_loader) * 32, attr_num))

    for i, _ in enumerate(val_loader):
        print(f"\r{i+1}/{len(val_loader)}\t", end='')

        with torch.no_grad():
            input, target = _
            target = target.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)
            output = model(input)

            # maximum voting
            if type(output) == type(()) or type(output) == type([]):
                output = torch.max(torch.max(torch.max(output[0],output[1]),output[2]),output[3])

            batch_size = target.size(0)
            output = torch.sigmoid(output.data).cpu().numpy()

            # print(output[0])
            scores[tol:tol+batch_size] = output
            targets[tol:tol+batch_size] = target.cpu().numpy()

            tol = tol + batch_size

    scores = scores[:tol]
    targets = targets[:tol]
    max_mAs = np.empty(attr_num)
    max_thress = np.empty(attr_num)
    each_thress = np.empty((attr_num, 100))

    for attr in range(attr_num):
        max_mA = 0.0
        max_thres = -1
        half_mA = 0.0

        for thres in range(2, 98+2, 2):    # 0.3 ~ 0.7, Grid search
            score_thres = np.where(scores[:, attr] > (thres / 100), 1, 0)
            pos_tol_one = 0
            pos_cnt_one = 0
            neg_tol_one = 0
            neg_cnt_one = 0

            for jt in range(tol):
                if targets[jt][attr] == 1:
                    pos_tol_one = pos_tol_one + 1
                    if score_thres[jt] == 1:
                        pos_cnt_one = pos_cnt_one + 1
                if targets[jt][attr] == 0:
                    neg_tol_one = neg_tol_one + 1
                    if score_thres[jt] == 0:
                        neg_cnt_one = neg_cnt_one + 1

            if pos_tol_one == 0:
                pos_mA = 1.0
            else:
                pos_mA = 1.0*pos_cnt_one/pos_tol_one
            if neg_tol_one == 0:
                neg_mA = 1.0
            else:
                neg_mA = 1.0*neg_cnt_one/neg_tol_one
                
            cur_mA = (pos_mA + neg_mA) / 2.0
            each_thress[attr, thres] = cur_mA

            # print(f"\r{attr} {thres/100} {int(cur_mA*1000)/1000}\t", end='')

            if max_mA < cur_mA:
                max_mA = cur_mA
                max_thres = thres

            if thres == 50:
                half_mA = cur_mA

        if max_mA == 0.5:
            max_thres = 50

        max_mAs[attr] = max_mA
        max_thress[attr] = max_thres

        print(f"{attr} HALF: {int(half_mA*1000)/1000}, MAX: {int(max_mA*1000)/1000}({max_thres/100})")
    
    np.save(f"./model/jagalchi/max_thres_{args.start_epoch}", max_thress)
    np.save("./model/jagalchi/each_thres", each_thress)



    
    


if __name__ == '__main__':
    main()