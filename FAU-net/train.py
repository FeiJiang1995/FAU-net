import os
from torch import nn, optim
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from data import *
from net import *
from Eval import eval_factor, val_loss_ep, onehot, rev_oneh
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = 'params/decy1/LR_0.000010/tranlr.pth'
data_train_path = r'VOC2007/train'
data_val_path = r'VOC2007/val'
save_path = 'train_image'
save_path2 = 'train_image2'
decy_list = [0, 1]  # 0 denotes not decy; 1 denotes decy
LR_list = [0.001000, 0.000100, 0.000010, 0.000001]
for decy in decy_list:
    for LR in LR_list:

        history_epoch = []
        history_step = []
        train_loss_step = 0
        train_time = 0

        train_loader = DataLoader(MyDataset(data_train_path), batch_size=1, shuffle=True)
        val_loader = DataLoader(MyDataset(data_val_path), batch_size=1, shuffle=True)
        steps_1epoch = len(train_loader)
        net = FAUnet(3, 4).to(device)
        if os.path.exists(weight_path):
            net.load_state_dict(torch.load(weight_path))
            print('successful load weight！')
        else:
            print('not successful load weight')

        opt = optim.Adam(net.parameters(), lr=LR)
        if decy == 1:
            sch = ExponentialLR(opt, gamma=0.99)
        loss_fun = nn.CrossEntropyLoss()

        epoch = 0
        while epoch <= 499:
            if decy == 1:
                print(f"{epoch}-LR===>>{opt.state_dict()['param_groups'][0]['lr']}")
            train_loss_epoch = 0
            for i, (image, segment_image) in enumerate(train_loader):
                torch.cuda.synchronize()
                start = time.time()

                image, segment_image = image.to(device), segment_image.to(device)
                out_image = net(image)

                _image = image[0]
                _segment_image = segment_image[0]
                _out_image = rev_oneh(out_image)[0]

                segment_image = onehot(segment_image)
                segment_image = torch.topk(segment_image, 1, dim=1)[1].squeeze(1)
                train_loss = loss_fun(out_image, segment_image)

                train_loss_step = train_loss.item()
                train_loss_epoch += (train_loss.item() / steps_1epoch)

                opt.zero_grad()
                train_loss.backward()
                opt.step()

                torch.cuda.synchronize()
                end = time.time()
                train_time += (end - start)

                history_step.append([train_loss_step, train_time])

                if i + 1 == steps_1epoch:
                    train_pa, train_mpa, train_miou, train_fwiou = eval_factor(net, train_loader)
                    print(
                        f'{epoch}-train_pa-train_mpa-train_miou-train_fwiou===>>{train_pa, train_mpa, train_miou, train_fwiou}')
                    val_pa, val_mpa, val_miou, val_fwiou = eval_factor(net, val_loader)
                    print(f'{epoch}-val_pa-val_mpa-val_miou-val_fwiou===>>{val_pa, val_mpa, val_miou, val_fwiou}')
                    val_loss_epoch = val_loss_ep(net, val_loader)
                    print(f'{epoch}-train_loss_epoch-val_loss_epoch===>>{train_loss_epoch, val_loss_epoch,}')

                    history_epoch.append(
                        [train_loss_epoch, val_loss_epoch, train_pa, train_mpa, train_miou, train_fwiou, val_pa,
                         val_mpa,
                         val_miou,
                         val_fwiou])
                    history_epoch_np = np.array(history_epoch)
                    np.save('logs/decy' + str(decy) + '/LR_' + str('%f' % LR) + '/history_epoch.npy', history_epoch_np)

                    history_step_np = np.array(history_step)
                    np.save('logs/decy' + str(decy) + '/LR_' + str('%f' % LR) + '/history_step.npy', history_step_np)

                if i % 1 == 0:
                    print(f'{epoch}-{i}-train_loss_decy_LR===>>{train_loss.item(), decy, LR}')

                img = torch.stack([_image, _segment_image, _out_image], dim=0)
                save_image(img, f'{save_path}/{i}.png')
                save_image(_out_image, f'{save_path2}/{i}.png')

            if epoch % 10 == 0:
                torch.save(net.state_dict(),
                           'params/decy' + str(decy) + '/LR_' + str('%f' % LR) + '/Unet' + str(epoch) + '.pth')
            if decy == 1:
                sch.step()
            epoch += 1
