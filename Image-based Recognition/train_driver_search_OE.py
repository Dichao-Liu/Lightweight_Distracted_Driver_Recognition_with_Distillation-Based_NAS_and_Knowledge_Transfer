from __future__ import print_function
import os
import logging
import torch.optim as optim
from models.pyconvhgresnet import PyConv2 as PyConv
from kd_losses import *

from utils import *


class BasicLayer(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, kernel_size_, group=1, group_=4):
        super(BasicLayer, self).__init__()
        self.convs = nn.Sequential(
            PyConv(in_planes, out_planes, pyconv_kernels=[kernel_size, kernel_size_], stride=1, pyconv_groups=[group, group_]),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class Cells1(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.Parameter(torch.ones(4, 1))

        self.cell1 = BasicLayer(3, 32, kernel_size=11, kernel_size_=7, group=1, group_=1)

        self.cell2 = BasicLayer(3, 32, kernel_size=11, kernel_size_=5, group=1, group_=1)

        self.cell3 = BasicLayer(3, 32, kernel_size=11, kernel_size_=3, group=1, group_=1)

        self.cell4 = BasicLayer(3, 32, kernel_size=11, kernel_size_=1, group=1, group_=1)

    def forward(self, x):
        x1 = self.cell1(x)
        x2 = self.cell2(x)
        x3 = self.cell3(x)
        x4 = self.cell4(x)

        p = torch.nn.functional.softmax(self.params, dim=0)

        x = x1*p[0] + x2*p[1] + x3*p[2] + x4*p[3]
        return x


class Cells1_pool(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.Parameter(torch.ones(2, 1))

        self.cell1 = nn.AvgPool2d(2, stride=2)

        self.cell2 = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x1 = self.cell1(x)
        x2 = self.cell2(x)
        p = torch.nn.functional.softmax(self.params, dim=0)
        x = x1*p[0] + x2*p[1]
        return x


class Cells2(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.Parameter(torch.ones(9, 1))

        self.cell1 = BasicLayer(32, 64, kernel_size=9, kernel_size_=5, group=1, group_=1)
        self.cell2 = BasicLayer(32, 64, kernel_size=9, kernel_size_=5, group=1, group_=2)
        self.cell3 = BasicLayer(32, 64, kernel_size=9, kernel_size_=5)

        self.cell4 = BasicLayer(32, 64, kernel_size=9, kernel_size_=3, group=1, group_=1)
        self.cell5 = BasicLayer(32, 64, kernel_size=9, kernel_size_=3, group=1, group_=2)
        self.cell6 = BasicLayer(32, 64, kernel_size=9, kernel_size_=3)

        self.cell7 = BasicLayer(32, 64, kernel_size=9, kernel_size_=1, group=1, group_=1)
        self.cell8 = BasicLayer(32, 64, kernel_size=9, kernel_size_=1, group=1, group_=2)
        self.cell9 = BasicLayer(32, 64, kernel_size=9, kernel_size_=1)

    def forward(self, x):
        x1 = self.cell1(x)
        x2 = self.cell2(x)
        x3 = self.cell3(x)
        x4 = self.cell4(x)
        x5 = self.cell5(x)
        x6 = self.cell6(x)
        x7 = self.cell7(x)
        x8 = self.cell8(x)
        x9 = self.cell9(x)

        p = torch.nn.functional.softmax(self.params, dim=0)

        x = x1*p[0] + x2*p[1] + x3*p[2] + x4*p[3] + x5*p[4] + x6*p[5] + x7*p[6] + x8*p[7] + x9*p[8]
        return x

class Cells2_pool(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.Parameter(torch.ones(2, 1))

        self.cell1 = nn.AvgPool2d(2, stride=2)

        self.cell2 = nn.MaxPool2d(2, stride=2)


    def forward(self, x):
        x1 = self.cell1(x)
        x2 = self.cell2(x)
        p = torch.nn.functional.softmax(self.params, dim=0)
        x = x1*p[0] + x2*p[1]
        return x


class Cells3(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.Parameter(torch.ones(6, 1))

        self.cell1 = BasicLayer(64, 128, kernel_size=5, kernel_size_=3, group=1, group_=1)
        self.cell2 = BasicLayer(64, 128, kernel_size=5, kernel_size_=3, group=1, group_=2)
        self.cell3 = BasicLayer(64, 128, kernel_size=5, kernel_size_=3)

        self.cell4 = BasicLayer(64, 128, kernel_size=5, kernel_size_=1, group=1, group_=1)
        self.cell5 = BasicLayer(64, 128, kernel_size=5, kernel_size_=1, group=1, group_=2)
        self.cell6 = BasicLayer(64, 128, kernel_size=5, kernel_size_=1)


    def forward(self, x):
        x1 = self.cell1(x)
        x2 = self.cell2(x)
        x3 = self.cell3(x)
        x4 = self.cell4(x)
        x5 = self.cell5(x)
        x6 = self.cell6(x)

        p = torch.nn.functional.softmax(self.params, dim=0)

        x = x1*p[0] + x2*p[1] + x3*p[2] + x4*p[3] + x5*p[4] + x6*p[5]
        return x

class Cells3_pool(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.Parameter(torch.ones(2, 1))

        self.cell1 = nn.AvgPool2d(2, stride=2)

        self.cell2 = nn.MaxPool2d(2, stride=2)


    def forward(self, x):
        x1 = self.cell1(x)
        x2 = self.cell2(x)
        p = torch.nn.functional.softmax(self.params, dim=0)
        x = x1*p[0] + x2*p[1]
        return x



class Cells4(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.Parameter(torch.ones(3, 1))

        self.cell1 = BasicLayer(128, 256, kernel_size=3, kernel_size_=1, group=1, group_=1)
        self.cell2 = BasicLayer(128, 256, kernel_size=3, kernel_size_=1, group=1, group_=2)
        self.cell3 = BasicLayer(128, 256, kernel_size=3, kernel_size_=1)

    def forward(self, x):
        x1 = self.cell1(x)
        x2 = self.cell2(x)
        x3 = self.cell3(x)

        p = torch.nn.functional.softmax(self.params, dim=0)

        x = x1*p[0] + x2*p[1] + x3*p[2]
        return x

class Cells4_pool(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.Parameter(torch.ones(3, 1))

        self.cell1 = nn.AvgPool2d(2, stride=2)

        self.cell2 = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x1 = self.cell1(x)
        x2 = self.cell2(x)
        p = torch.nn.functional.softmax(self.params, dim=0)
        x = x1*p[0] + x2*p[1]
        return x


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


class Features(nn.Module):
    def __init__(self, net_layers):
        super(Features, self).__init__()
        self.net_layers = net_layers

    def forward(self, x):
        x = self.net_layers[0](x)
        x = self.net_layers[1](x)
        x = self.net_layers[2](x)
        x = self.net_layers[3](x)
        x = self.net_layers[4](x)
        x1 = self.net_layers[5](x)
        x2 = self.net_layers[6](x1)
        x3 = self.net_layers[7](x2)
        return x1, x2, x3


class Search_Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1 = Cells1()
        self.L1P = Cells1_pool()
        self.L2 = Cells2()
        self.L2P = Cells2_pool()
        self.L3 = Cells3()
        self.L3P = Cells3_pool()
        self.L4 = Cells4()
        self.L4P = Cells4_pool()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.L1(x)
        x = self.L1P(x)
        x = self.L2(x)
        x = self.L2P(x)
        x = self.L3(x)
        x = self.L3P(x)
        x = self.L4(x)
        x = self.L4P(x)
        x = self.pool(x).view(x.shape[0], -1)
        x = self.classifier(x)
        return x


def train(nb_epoch, batch_size, store_name, resume=False, start_epoch=0, model_path=None):
    criterionKD = Logits()
    _logger = logging.getLogger('train')

    exp_dir = store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)

    use_cuda = torch.cuda.is_available()
    print(use_cuda)


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
    net_teacher = torch.load("save_teacher_OE/model.pth")



    net = Search_Wrapper()


    net = torch.nn.DataParallel(net, device_ids=[0,1]).cuda()
    net_teacher = torch.nn.DataParallel(net_teacher, device_ids=[0]).cuda()

    device = torch.device("cuda")
    net.to(device)


    CELoss = nn.CrossEntropyLoss()


    optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9, weight_decay=5e-4)

    max_val_acc = 0
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002]
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

            with torch.no_grad():
                _, _, _, output_teacher = net_teacher(inputs)

            optimizer.zero_grad()
            output = net(inputs)
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
            val_acc, val_loss = test(net, CELoss, 16)
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                net.cpu()
                torch.save(net, './' + store_name + '/model.pth')
                net.to(device)
            with open(exp_dir + '/results_test.txt', 'a') as file:
                file.write('IteratiFon %d, test_acc = %.5f, test_loss = %.6f\n' % (
                    epoch, val_acc, val_loss))
        else:
            net.cpu()
            torch.save(net, './' + store_name + '/model.pth')
            net.to(device)


if __name__ == '__main__':
    save_path = 'save_search_OE'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    train(nb_epoch=300,  # number of epoch
          batch_size=32, # batch size
          store_name=save_path,  # folder for output
          resume=False,  # resume training from checkpoint
          start_epoch=0,  # the start epoch number when you resume the training
          model_path='')  # the saved model where you want to resume the training
