import argparse
import os
import csv
from numpy.core.numeric import flatnonzero
import xlwt
import random
import time
import warnings
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
import torchvision.datasets as datasets
import shutil
from torchvision.transforms import ToPILImage
from collections import Counter
import models
import torch.nn.functional as F
import scipy.io as io
#当前环境没有 tensorboardX包
# from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from utils import *

start_time = time.time()

#获取模块中所有可调用的公共函数名列表
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Malimg Training')
parser.add_argument('--dataset', default='Malimg', help='dataset setting')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet_comb',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet32)')
parser.add_argument('--resize', default="224", type=int, help='resize size')
parser.add_argument('--loss_type', default="CE", type=str, help='loss type')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=1, type=float, help='imbalance factor')
parser.add_argument('--train_rule', default='None', type=str, help='data sampling strategy for train loader')
parser.add_argument('--rand_number', default=42, type=int, help='fix random number for data sampling')
parser.add_argument('--exp_str', default='0', type=str, help='number to indicate which experiment it is')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--stage', default=160, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=1, type=int,
                    help='GPU id to use.')
parser.add_argument('--root_log', type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')
best_acc1 = 0
best_acc1_train = 0
parser.add_argument('--lam', default=0.25, type=float, help='[0.25, 0.5, 0.75, 1.0]')
parser.add_argument('--alpha', default=0.25, type=float, help='[0.25, 0.5, 0.75, 1.0]')


def main():
    args = parser.parse_args()
    for arg in vars(args):
        print("{}={}".format(arg, getattr(args, arg)))
    args.store_name = '_'.join(
        [str(args.epochs), args.dataset, args.arch, str(args.resize),
         str(args.imb_factor), args.exp_str, "best"])
    prepare_folders(args)
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    #获取当前设备上可用的GPU数量
    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1, best_acc1_train
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    num_classes = 25
    use_norm = True if args.loss_type == 'LDAM' else False  #默认是CE损失函数
    model = models.__dict__[args.arch](num_classes=num_classes, use_norm=use_norm)

    if args.gpu is not None: #在指定GPU上运行
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else: #没指定具体GPU，使用 DataParallel 函数将模型包装起来，这样模型就可以在所有可用的 GPU 上进行并行计算。
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, nesterov=args.nesterov,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    #加载模型训练的断点
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    # Data loading code
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 读取数据集
    # 数据集根目录
    # current_directory = os.path.dirname(__file__)
    # dataset_root = "/data/ch/tmp/pycharm_project_191/fsdr/malimg/malimg_dataset/bal_dataset"
    dataset_root = "/data/ch/tmp/pycharm_project_191/fsdr/malimg_dataset/Imb_dataset"

    # 创建ImageFolder对象
    train_dataset = ImageFolder(root=dataset_root + "/train_dataset", transform=transform_train)
    val_dataset = ImageFolder(root=dataset_root + "/val_dataset", transform=transform_val)

    # 统计每个类别的图像数量
    class_counts = Counter(train_dataset.targets)

    # 存储每个类别的图像数量
    img_num_list = []
    for class_idx in sorted(class_counts.keys()):
        class_name = train_dataset.classes[class_idx]
        count = class_counts[class_idx]
        img_num_list.append(count)

    # 打印每个类别的图像数量
    print('img_num_list:')
    print(img_num_list)
    print(len(img_num_list))
    # args.img_num_list = img_num_list

    # Val
    # 统计每个类别的图像数量
    class_counts_val = Counter(val_dataset.targets)

    # 存储每个类别的图像数量
    img_num_list_val = []
    for class_idx_val in sorted(class_counts_val.keys()):
        class_name_val = val_dataset.classes[class_idx_val]
        count_val = class_counts_val[class_idx_val]
        img_num_list_val.append(count_val)

    # 打印每个类别的图像数量
    print('img_num_list_val:')
    print(img_num_list_val)
    print(len(img_num_list_val))

    args.cls_num_list = img_num_list

    train_sampler = None

    #shuffle表示在每个 epoch 开始时是否打乱数据；num_workers表示用于数据加载的子进程数
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # init log for training
    # log_epoch：用于记录每个epoch结束后的日志信息，保存在log_epoch.csv文件中。
    # log_training：用于记录训练过程中的日志信息，保存在log_train.csv文件中。
    # log_testing：用于记录测试过程中的日志信息，保存在log_test.csv文件中。
    # log_eff：可能用于记录模型效率或其他指标的日志信息，保存在log_effall.csv文件中。
    # args.txt：保存了当前程序运行所用的参数信息，以便在需要时进行查看或复现实验结果。
    log_epoch = open(os.path.join(args.root_log, args.store_name, 'log_epoch.csv'), 'w')

    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    log_eff = open(os.path.join(args.root_log, args.store_name, 'log_effall.csv'), 'w')

    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))

    #创建了一个 TensorBoard 的 SummaryWriter，用于将日志数据写入到指定的目录下，通常用于可视化训练过程中的损失、准确率等信息。
    # tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    tf_writer = []#随意定义一个，不做任何处理

    #创建了一个 Excel 工作簿对象。
    workbook = xlwt.Workbook(encoding='utf-8')
    # worksheet
    worksheet = workbook.add_sheet('My Worksheet')

    train_propertype_best = None
    weights_old_best = None

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        ratio = epoch
        # First stage
        if epoch < 80: #160
            train_propertype = None
            weights_old = None
            train(train_loader, model, optimizer, epoch, args, log_training, tf_writer, ratio, log_epoch,
                  train_propertype, weights_old)

            acc1 = validate(train_loader, model, epoch, args, log_testing, tf_writer)
            is_best_train = acc1 > best_acc1_train
            if is_best_train:
                print('Change.')
                train_propertype_best = get_feature_mean(train_loader, model, args.cls_num_list).cuda()
                train_propertype_best = train_propertype_best.detach()
                weights_old_best = calculate_eff_weight(train_loader, model, args.cls_num_list, train_propertype_best, log_eff)
                weights_old_best = weights_old_best.detach()

            best_acc1_train = max(acc1, best_acc1_train)

        # Second stage
        else: #权重冻结
            # train_propertype = get_feature_mean(train_loader, model, args.cls_num_list).cuda()
            # train_propertype = train_propertype.detach()
            # weights_old = calculate_eff_weight(train_loader, model, args.cls_num_list, train_propertype, log_eff)
            # weights_old = weights_old.detach()
            train(train_loader, model, optimizer, epoch, args, log_training, tf_writer, ratio, log_epoch, train_propertype_best,
                  weights_old_best)


        if epoch >= 30: #150
            # evaluate on validation set
            acc1 = validate(val_loader, model, epoch, args, log_testing, tf_writer)
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            # tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
            output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
            print(output_best)
            # log_testing.write(output_best + '\n')
            # log_testing.flush()


def train(train_loader, model, optimizer, epoch, args, log, tf_writer, ratio, log_epoch, train_propertype, weights_old):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        features, output = model(input)
        if epoch < 80: #160
            loss = F.cross_entropy(output, target)
        else:
            # stage two reweighting
            loss = F.cross_entropy(output, target, weight=weights_old)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5,
                lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
            print(output)
            log.write(output + '\n')
            log.flush()

    # tf_writer.add_scalar('loss/train', losses.avg, epoch)
    # tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    # tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    # tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)


def get_feature_mean(imbalanced_train_loader, model, cls_num_list):
    model.eval()
    cls_num = len(cls_num_list)
    feature_mean_end = torch.zeros(cls_num, 4096)
    with torch.no_grad():
        for i, (input, target) in enumerate(imbalanced_train_loader):
            target = target.cuda()
            input = input.cuda()
            input_var = to_var(input, requires_grad=False)
            features, output = model(input_var)
            #这种操作通常用于在特定的情况下，我们希望保留特征张量的值，但不需要它们继续对后续计算和模型参数优化产生影响。
            # 通过使用 detach() 方法，可以有效地将特征张量从梯度计算中分离，使其成为一个普通的张量，而不是一个计算梯度的变量。
            #这种技术在某些情况下很有用，例如在特征提取阶段，我们需要保留特征张量的值以供其他用途，但不需要对这些特征进行梯度更新。这有助于节省内存和计算资源，同时保留我们需要的信息。
            features = features.detach()
            features = features.cpu().data.numpy()

            for out, label in zip(features, target):
                feature_mean_end[label] = feature_mean_end[label] + out

        img_num_list_tensor = torch.tensor(cls_num_list).unsqueeze(1)
        feature_mean_end = torch.div(feature_mean_end, img_num_list_tensor).detach()
        return feature_mean_end


def calculate_eff_weight(imbalanced_train_loader, model, cls_num_list, train_propertype, log_eff):
    model.eval()
    train_propertype = train_propertype.cuda()
    class_num = len(cls_num_list)
    eff_all = torch.zeros(class_num).float().cuda()
    with torch.no_grad():
        for i, (input, target) in enumerate(imbalanced_train_loader):
            target = target.cuda()
            input = input.cuda()
            input_var = to_var(input, requires_grad=False)
            features, output = model(input_var)
            mu = train_propertype[target].detach()  # batch_size x d
            feature_bz = (features.detach() - mu)  # Centralization
            index = torch.unique(target)  # class subset
            index2 = target.cpu().numpy()
            eff = torch.zeros(class_num).float().cuda()

            for i in range(len(index)):  # number of class
                index3 = torch.from_numpy(np.argwhere(index2 == index[i].item()))
                index3 = torch.squeeze(index3)
                feature_juzhen = feature_bz[index3].detach()
                if feature_juzhen.dim() == 1:
                    eff[index[i]] = 1
                else:
                    _matrixA_matrixB = torch.matmul(feature_juzhen, feature_juzhen.transpose(0, 1))
                    _matrixA_norm = torch.unsqueeze(torch.sqrt(torch.mul(feature_juzhen, feature_juzhen).sum(axis=1)),
                                                    1)
                    _matrixA_matrixB_length = torch.mul(_matrixA_norm, _matrixA_norm.transpose(0, 1))
                    _matrixA_matrixB_length[_matrixA_matrixB_length == 0] = 1
                    r = torch.div(_matrixA_matrixB, _matrixA_matrixB_length)  # R
                    num = feature_juzhen.size(0)
                    a = (torch.ones(1, num).float().cuda()) / num  # a_T
                    b = (torch.ones(num, 1).float().cuda()) / num  # a
                    c = torch.matmul(torch.matmul(a, r), b).float().cuda()  # a_T R a
                    eff[index[i]] = 1 / c
            eff_all = eff_all + eff
        weights = eff_all
        log_eff.write(str(eff_all) + '\n')
        log_eff.flush()
        weights = torch.where(weights > 0, 1 / weights, weights).detach()
        # weight
        fen_mu = torch.sum(weights)
        weights_new = weights / fen_mu
        weights_new = weights_new * class_num  # Eq.(14)
        # print(weights_new) #类别权重
        print('new weights')
        weights_new = weights_new.detach()

        return weights_new


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def validate(val_loader, model, epoch, args, log=None, tf_writer=None, flag='val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            features, output = model(input)
            # output = model(input)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
        #通过混淆矩阵来计算每个类别的准确率
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        output = ('Epoch: [{0}] {flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                  .format(epoch, flag=flag, top1=top1, top5=top5, loss=losses))
        out_cls_acc = '%s Class Accuracy: %s' % (
        flag, (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
        print(output)
        # print(out_cls_acc) #每个类别的准确率
        if log is not None:
            log.write(output + '\n')
            log.write(out_cls_acc + '\n')
            log.flush()

        # tf_writer.add_scalar('loss/test_' + flag, losses.avg, epoch)
        # tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg, epoch)
        # tf_writer.add_scalar('acc/test_' + flag + '_top5', top5.avg, epoch)
        # tf_writer.add_scalars('acc/test_' + flag + '_cls_acc', {str(i): x for i, x in enumerate(cls_acc)}, epoch)

    return top1.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 5:
        lr = args.lr * float(epoch + 1) / 5
    else:
        lr = args.lr * ((0.1 ** int(epoch >= 40)) * (0.1 ** int(epoch >= 60)) * (0.1 ** int(epoch >= 80)))
        # lr = args.lr * ((0.1 ** int(epoch >= 40)) * (0.1 ** int(epoch >= 60)) * (0.1 ** int(epoch >= 80)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
    end_time = time.time()
    run_time_sec = end_time - start_time
    hours = run_time_sec // 3600
    minutes = (run_time_sec % 3600) // 60
    seconds = run_time_sec % 60
    output_str = f"{hours}时{minutes}分{seconds}秒"
    print("程序运行时间：", output_str)