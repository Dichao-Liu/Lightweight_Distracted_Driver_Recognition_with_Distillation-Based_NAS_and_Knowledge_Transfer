from __future__ import print_function
import os
from PIL import Image
import argparse
import logging
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import timm
import os
from kd_losses import *


from utils_distill import *
from models.pyconvhgresnet import PyConv2 as PyConv




parser = argparse.ArgumentParser(description='train kd')
parser.add_argument('--kd_mode', type=str, required=True, help='mode of knowledge distillation')
parser.add_argument('--T', type=float, default=4.0, help='temperature for ST')
parser.add_argument('--p', type=float, default=2.0, help='power for AT')
parser.add_argument('--w_dist', type=float, default=25.0, help='weight for RKD distance')
parser.add_argument('--w_angle', type=float, default=50.0, help='weight for RKD angle')
parser.add_argument('--m', type=float, default=2.0, help='margin for AB')
parser.add_argument('--gamma', type=float, default=0.4, help='gamma in Gaussian RBF for CC')
parser.add_argument('--P_order', type=int, default=2, help='P-order Taylor series of Gaussian RBF for CC')
parser.add_argument('--w_irg_vert', type=float, default=0.1, help='weight for IRG vertex')
parser.add_argument('--w_irg_edge', type=float, default=5.0, help='weight for IRG edge')
parser.add_argument('--w_irg_tran', type=float, default=5.0, help='weight for IRG transformation')



args, unparsed = parser.parse_known_args()


class BasicLayer(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, kernel_size_, group=1, group_=4):
        super(BasicLayer, self).__init__()
        self.convs = nn.Sequential(
            PyConv(in_planes, out_planes, pyconv_kernels=[kernel_size, kernel_size_], stride=1, pyconv_groups=[group, group_]),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x = self.convs(x)
        x = self.pool(x)

        return x


class Searched_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            BasicLayer(3, 32, kernel_size=11, kernel_size_=3, group=1, group_=1),
            BasicLayer(32, 64, kernel_size=9, kernel_size_=5, group=1, group_=1),
            BasicLayer(64, 128, kernel_size=5,  kernel_size_=3, group=1, group_=1),
            BasicLayer(128, 256, kernel_size=3,  kernel_size_=1, group=1, group_=1)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.convs(x)
        # x = self.dropout(x)
        x = self.pool(x).view(x.shape[0], -1)
        x = self.classifier(x)
        return x




class Features(nn.Module):
    def __init__(self, net_layers):
        super(Features, self).__init__()
        self.net_layer_0 = nn.Sequential(net_layers[0])
        self.net_layer_1 = nn.Sequential(net_layers[1])
        self.net_layer_2 = nn.Sequential(net_layers[2])
        self.net_layer_3 = nn.Sequential(net_layers[3])
        self.net_layer_4 = nn.Sequential(*net_layers[4])
        self.net_layer_5 = nn.Sequential(*net_layers[5])
        self.net_layer_6 = nn.Sequential(*net_layers[6])
        self.net_layer_7 = nn.Sequential(*net_layers[7])


    def forward(self, x):
        x = self.net_layer_0(x)
        x = self.net_layer_1(x)
        x = self.net_layer_2(x)
        x = self.net_layer_3(x)
        x = self.net_layer_4(x)
        x1 = self.net_layer_5(x)
        x2 = self.net_layer_6(x1)
        x3 = self.net_layer_7(x2)
        return x1, x2, x3



class Network_Wrapper(nn.Module):
    def __init__(self, net_layers):
        super().__init__()
        self.Features = Features(net_layers)

        self.max_pool1 = nn.MaxPool2d(kernel_size=28, stride=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=14, stride=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=7, stride=1)

        self.conv_block1 = nn.Sequential(
            BasicConv(512, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, 10)
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(1024, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, 10),
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(2048, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, 10),
        )

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(1024 * 3),
            nn.Linear(1024 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x1, x2, x3 = self.Features(x)

        x1_ = self.conv_block1(x1)
        x1_ = self.max_pool1(x1_)
        x1_f = x1_.view(x1_.size(0), -1)

        x1_c = self.classifier1(x1_f)


        x2_ = self.conv_block2(x2)
        x2_ = self.max_pool2(x2_)
        x2_f = x2_.view(x2_.size(0), -1)
        x2_c = self.classifier2(x2_f)

        x3_ = self.conv_block3(x3)
        x3_ = self.max_pool3(x3_)
        x3_f = x3_.view(x3_.size(0), -1)
        x3_c = self.classifier3(x3_f)

        x_c_all = torch.cat((x1_f, x2_f, x3_f), -1)
        x_c_all = self.classifier_concat(x_c_all)

        return x1_c, x2_c, x3_c, x_c_all





def train(nb_epoch, batch_size, store_name, resume=False, start_epoch=0, model_path=None):
    # setup output

    if args.kd_mode == 'logits':
        criterionKD = Logits()
    elif args.kd_mode == 'st':
        criterionKD = SoftTarget(args.T)
    elif args.kd_mode == 'at':
        criterionKD = AT(args.p)
    elif args.kd_mode == 'fitnet':
        criterionKD = Hint()
    elif args.kd_mode == 'nst':
        criterionKD = NST()
    elif args.kd_mode == 'pkt':
        criterionKD = PKTCosSim()
    elif args.kd_mode == 'fsp':
        criterionKD = FSP()
    elif args.kd_mode == 'rkd':
        criterionKD = RKD(args.w_dist, args.w_angle)
    elif args.kd_mode == 'ab':
        criterionKD = AB(args.m)
    elif args.kd_mode == 'sp':
        criterionKD = SP()
    elif args.kd_mode == 'sobolev':
        criterionKD = Sobolev()
    elif args.kd_mode == 'cc':
        criterionKD = CC(args.gamma, args.P_order)
    elif args.kd_mode == 'lwm':
        criterionKD = LwM()
    elif args.kd_mode == 'irg':
        criterionKD = IRG(args.w_irg_vert, args.w_irg_edge, args.w_irg_tran)
    else:
        raise Exception('Invalid kd mode...')



    exp_dir = store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)

    use_cuda = torch.cuda.is_available()
    print(use_cuda)


    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Scale((256, 256)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = torchvision.datasets.ImageFolder(root='<the path to>/train',
                                                transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    net_teacher = torch.load("save_teacher/model.pth")
    net = Searched_Net()

    print('Model %s created, param count: %d' %
          ('Created_model', sum([m.numel() for m in net.parameters()])))


    netp_teacher = torch.nn.DataParallel(net_teacher).cuda()
    netp = torch.nn.DataParallel(net).cuda()
    device = torch.device("cuda")
    cudnn.benchmark = True

    CELoss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9, weight_decay=5e-4)

    max_val_acc = 0
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002]
    for epoch in range(start_epoch, nb_epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        idx = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            idx = batch_idx
            if inputs.shape[0] < batch_size:
                continue
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)

            # update learning rate
            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])

            if args.kd_mode in ['sobolev', 'lwm']:
                inputs.requires_grad = True

            with torch.no_grad():
                _, _, _, output_teacher = netp_teacher(inputs)


            optimizer.zero_grad()
            output = netp(inputs)

            if args.kd_mode in ['sobolev']:
                loss = criterionKD(output, output_teacher, inputs, targets)
            else:
                loss = 0.7*criterionKD(output, output_teacher) + 0.3 * CELoss(output, targets)


            loss.backward()
            optimizer.step()

            #  training log
            _, predicted = torch.max(output.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            train_loss += loss.item()

            if batch_idx % 50 == 0 or batch_idx == trainloader.__len__() - 1:
                print(
                    'Step: %d | Loss1: %.3f | Acc: %.3f%% (%d/%d)' % (
                        batch_idx, train_loss / (batch_idx + 1),
                        100. * float(correct) / total, correct, total))

        train_acc = 100. * float(correct) / total
        train_loss = train_loss / (idx + 1)
        with open(exp_dir + '/results_train.txt', 'a') as file:
            file.write(
                'Iteration %d | train_acc = %.5f | train_loss = %.5f |\n' % (
                    epoch, train_acc, train_loss))

        if epoch < 5 or epoch >= 50:
            val_acc, val_loss = test(net, netp_teacher, CELoss, 16)
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                net.cpu()
                torch.save(net, './' + store_name + '/model.pth')
                net.to(device)
            with open(exp_dir + '/results_test.txt', 'a') as file:
                file.write('Iteration %d, test_acc = %.5f, test_loss = %.6f\n' % (
                    epoch, val_acc, val_loss))
        else:
            net.cpu()
            torch.save(net, './' + store_name + '/model.pth')
            net.to(device)

if __name__ == '__main__':
    save_path = 'save_transfer_' + args.kd_mode
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    train(nb_epoch=300,  # number of epoch
          batch_size=32,  # batch size
          store_name=save_path,  # folder for output
          resume=False,  # resume training from checkpoint
          start_epoch=0,  # the start epoch number when you resume the training
          model_path='')  # the saved model where you want to resume the training
