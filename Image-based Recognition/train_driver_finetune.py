from __future__ import print_function
import os
import logging
import torch.optim as optim
from models.pyconvhgresnet import PyConv2 as PyConv
from utils import *
import argparse

parser = argparse.ArgumentParser(description='Finetune')
parser.add_argument('--OE', action='store_true', default=False,
                    help='Whether to use OE')

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
        x = self.pool(x).view(x.shape[0], -1)
        x = self.classifier(x)
        return x



def train(nb_epoch, batch_size, store_name, resume=False, start_epoch=0, model_path=None):
    _logger = logging.getLogger('train')
    # setup output
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

    if args.OE:
        net = torch.load("save_transfer_OE_logits/model.pth")
    else:
        net = torch.load("save_transfer_logits/model.pth")

    print('Model %s created, param count: %d' %
          ('Created_model', sum([m.numel() for m in net.parameters()])))


    netp = torch.nn.DataParallel(net, device_ids=[0,1])

    device = torch.device("cuda")
    net.to(device)


    CELoss = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9, weight_decay=5e-4)

    max_val_acc = 0
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002]
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

            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])

            optimizer.zero_grad()
            output = netp(inputs)
            loss = CELoss(output, targets)
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
                file.write('Iteration %d, test_acc = %.5f, test_loss = %.6f\n' % (
                    epoch, val_acc, val_loss))
        else:
            net.cpu()
            torch.save(net, './' + store_name + '/model.pth')
            net.to(device)


if __name__ == '__main__':
    if args.OE:
        save_path = "save_finetune_OE"
    else:
        save_path = "save_finetune"

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    train(nb_epoch=300,  # number of epoch
          batch_size=32,  # batch size
          store_name=save_path,  # folder for output
          resume=False,  # resume training from checkpoint
          start_epoch=0,  # the start epoch number when you resume the training
          model_path='')  # the saved model where you want to resume the training
