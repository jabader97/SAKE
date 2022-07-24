import argparse
import os
import time
import pickle
from senet import cse_resnet50
from Sketchy import SketchyDataset
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import datetime
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from scipy.spatial.distance import cdist
import warnings
import pretrainedmodels
import torch.nn.functional as F
from ResnetModel import CSEResnetModel_KD, CSEResnetModel_KDHashing
import utils
import wandb
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score
from test_cse_resnet_tuberlin_zeroshot import eval_precision, VOCap, eval_AP_inner

# warnings.filterwarnings("error")

model_names = sorted(name for name in pretrainedmodels.__dict__
                     if name.islower() and not name.startswith("__"))

parser = argparse.ArgumentParser(description='PyTorch ResNet Model for Sketchy mAP Testing')
parser.add_argument('--arch', '-a', metavar='ARCH', default='cse_resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: se_resnet50)')
parser.add_argument('--num_classes', metavar='N', type=int, default=100,
                    help='number of classes (default: 100)')
parser.add_argument('--num_hashing', metavar='N', type=int, default=64,
                    help='number of hashing dimension (default: 64)')
parser.add_argument('--batch_size', default=50, type=int, metavar='N',
                    help='number of samples per batch')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')

parser.add_argument('--resume_dir',
                    default='../cse_resnet50/checkpoint/sketchy_kd1kdneg03sake1_f64/',
                    type=str, metavar='PATH',
                    help='dir of model checkpoint (default: none)')

parser.add_argument('--resume_file',
                    default='model_best.pth.tar',
                    type=str, metavar='PATH',
                    help='file name of model checkpoint (default: none)')

parser.add_argument('--ems-loss', dest='ems_loss', action='store_true',
                    help='use ems loss for the training')
parser.add_argument('--precision', action='store_true', help='report precision@100')
parser.add_argument('--pretrained', action='store_true', help='use pretrained model')
parser.add_argument('--zero_version', metavar='VERSION', default='zeroshot', type=str,
                    help='zeroshot version for training and testing (default: zeroshot)')
parser.add_argument('--log_online', action='store_true',
                    help='Flag. If set, run metrics are stored online in addition to offline logging. Should generally be set.')
parser.add_argument('--wandb_key', default='<your_api_key_here>', type=str, help='API key for W&B.')
parser.add_argument('--project', default='Sample_Project', type=str,
                    help='Name of the project - relates to W&B project names. In --savename default setting part of the savename.')
parser.add_argument('--group', default='Sample_Group', type=str, help='Name of the group - relates to W&B group names - all runs with same setup but different seeds are logged into one group. \
                                                                                           In --savename default setting part of the savename.')
parser.add_argument('--savename', default='group_plus_seed', type=str,
                    help='Run savename - if default, the savename will comprise the project and group name (see wandb_parameters()).')
parser.add_argument('--path_aux', type=str, default=os.getcwd())


def main():
    global args
    args = parser.parse_args()
    if args.savename == 'group_plus_seed':
        if args.log_online:
            args.savename = args.group
        else:
            args.savename = ''
    if args.zero_version == 'zeroshot2':
        args.num_classes = 104

    print('prepare SBIR features using saved model')
    apsall, aps200, prec100, prec200 = prepare_features()
    if args.log_online:
        valid_data = {"test_aps@all": apsall, "test_aps@200": aps200, "test_prec@100": prec100,
                      "test_prec@200": prec200}
        wandb.log(valid_data)


def prepare_features():
    # create model
    # model = cse_resnet50(num_classes = args.num_classes, pretrained=None)
    # model = CSEResnetModel_KD(args.arch, args.num_classes, ems=args.ems_loss)
    model = CSEResnetModel_KDHashing(args.arch, args.num_hashing, args.num_classes)
    # model.cuda()
    model = nn.DataParallel(model)
    print(str(datetime.datetime.now()) + ' model inited.')

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    # resume from a checkpoint
    if args.resume_file:
        resume = os.path.join(args.resume_dir, args.resume_file)
    else:
        resume = os.path.join(args.resume_dir, 'model_best.pth.tar')

    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        args.start_epoch = checkpoint['epoch']

        save_dict = checkpoint['state_dict']
        model_dict = model.state_dict()

        trash_vars = [k for k in save_dict.keys() if k not in model_dict.keys()]
        print('trashed vars from resume dict:')
        print(trash_vars)

        resume_dict = {k: v for k, v in save_dict.items() if k in model_dict}
        # resume_dict['module.linear.cpars'] = save_dict['module.linear.weight']

        model_dict.update(resume_dict)
        model.load_state_dict(model_dict)

        # model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))
        # return

    cudnn.benchmark = True

    # load data
    immean = [0.485, 0.456, 0.406]  # RGB channel mean for imagenet
    imstd = [0.229, 0.224, 0.225]

    transformations = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize([224, 224]),
                                          transforms.ToTensor(),
                                          transforms.Normalize(immean, imstd)])
    # image
    sketchy_zero_ext = SketchyDataset(split='zero', version='all_photo', zero_version=args.zero_version, \
                                      transform=transformations, aug=False)

    zero_loader_ext = DataLoader(dataset=sketchy_zero_ext, \
                                 batch_size=args.batch_size, shuffle=False, num_workers=0)
    # sketch
    sketchy_zero = SketchyDataset(split='zero', zero_version=args.zero_version, transform=transformations, aug=False)
    zero_loader = DataLoader(dataset=sketchy_zero, batch_size=args.batch_size, shuffle=False, num_workers=0)

    return get_features(zero_loader, zero_loader_ext, model)


def get_features(data_loader_sketch, data_loader_image, model):
    # Switch to test mode
    model.eval()

    # Start counting time
    for i, (sk, cls_sk, ti) in enumerate(data_loader_sketch):
        if torch.cuda.is_available():
            sk = sk.cuda()

        # Sketch embedding into a semantic space
        sk_em = get_sketch_embeddings(sk, model)

        # Accumulate sketch embedding
        if i == 0:
            acc_sk_em = sk_em.cpu().data.numpy()
            acc_cls_sk = cls_sk
        else:
            acc_sk_em = np.concatenate((acc_sk_em, sk_em.cpu().data.numpy()), axis=0)
            acc_cls_sk = np.concatenate((acc_cls_sk, cls_sk), axis=0)

    for i, (im, cls_im, ti) in enumerate(data_loader_image):

        if torch.cuda.is_available():
            im = im.cuda()

        # Image embedding into a semantic space
        im_em = get_image_embeddings(im, model)

        # Accumulate sketch embedding
        if i == 0:
            acc_im_em = im_em.cpu().data.numpy()
            acc_cls_im = cls_im
        else:
            acc_im_em = np.concatenate((acc_im_em, im_em.cpu().data.numpy()), axis=0)
            acc_cls_im = np.concatenate((acc_cls_im, cls_im), axis=0)

    sim_euc = np.exp(-cdist(acc_sk_em, acc_im_em, metric='euclidean'))

    # similarity of classes or ground truths
    # Multiplied by 1 for boolean to integer conversion
    str_sim = (np.expand_dims(acc_cls_sk, axis=1) == np.expand_dims(acc_cls_im, axis=0)) * 1

    apsall = utils.apsak(sim_euc, str_sim)
    aps200 = utils.apsak(sim_euc, str_sim, k=200)
    prec100, _ = utils.precak(sim_euc, str_sim, k=100)
    prec200, _ = utils.precak(sim_euc, str_sim, k=200)

    return apsall, aps200, prec100, prec200


def get_sketch_embeddings(sk, model):
    tag = torch.zeros(sk.size()[0], 1)
    if torch.cuda.is_available():
        tag = tag.cuda()
    features = model.module.original_model.features(sk, tag)
    features = model.module.original_model.hashing(features)
    features = F.normalize(features)
    return features


def get_image_embeddings(im, model):
    tag = torch.ones(im.size()[0], 1)
    if torch.cuda.is_available():
        tag = tag.cuda()
    features = model.module.original_model.features(im, tag)
    features = model.module.original_model.hashing(features)
    features = F.normalize(features)
    return features


if __name__ == '__main__':
    main()

