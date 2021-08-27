from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as NativeDDP
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
# import torchsummary
from torchvision import datasets, transforms
import torchvision.utils

import os, time, logging, yaml
import numpy as np
import mlp_mixer_slr.admm_code.admm as admm
# import admm_code.admm as admm

import warnings

from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint,\
    convert_splitbn_model, model_parameters
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
    wandb.init(project='mlp-mixer-slr', entity='zhoushanglin100')
except ImportError: 
    has_wandb = False

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

torch.cuda.empty_cache()

###################################################################
### check; Todo: modify config

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch SLR ImageNet Training')
               
# ---------------------- arg for admm train ----------------------------------
parser.add_argument('--optimization', type=str, default='savlr',
                    help='optimization type: [savlr, admm]')
parser.add_argument('--load-baseline-model', type=str, default="Baseline_NIN_model_best.pth.tar", 
                    help='checkpoint to start from (baseline model)')
parser.add_argument('--admm-train', action='store_true', default=False,
                    help='Choose admm training')
parser.add_argument('--masked-retrain', action='store_true', default=False,
                    help='Choose masked retraining')
parser.add_argument('--combine-progressive', action='store_true', default=False,
                    help='Choose combine progressive')
parser.add_argument('--config-file', type=str, default='config_mlp_0.5', 
                    help="prune config file")
parser.add_argument('--ext', type=str, default='', 
                    help="extension for saved file")
parser.add_argument('--admm-epoch', type=int, default=10, 
                    help="how often we do admm update")
parser.add_argument('--rho', type=float, default=0.1, 
                    help="define rho for ADMM")
parser.add_argument('--rho-num', type=int, default=1, 
                    help="define how many rohs for ADMM training")
parser.add_argument('--M', type=int, default=300, metavar='N',
                    help='SLR parameter M ')
parser.add_argument('--r', type=float, default=0.1, metavar='N',
                    help='SLR parameter r ')
parser.add_argument('--initial-s', type=float, default=0.01, metavar='N',
                    help='SLR parameter initial stepsize')
parser.add_argument('--retrain-epoch', type=int, default=10, metavar='N',
                    help='Number of retrain epochs')
parser.add_argument('--admmtrain-acc', type=float, default=76.08, metavar='N',
                    help='SLR trained best acc for saved model')
parser.add_argument('--sparsity-type', type=str, default='irregular',
                    help='sparsity type: [irregular,column,channel,filter,pattern,random-pattern]')
parser.add_argument('-p', '--print-freq', default=100, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--num-gpu', type=int, default=3,
                    help='Number of GPUS to use')
# -------------------------------------------------------------------
# Dataset / Model parameters
parser.add_argument('data_dir', metavar='DIR', default='/data',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='imagenet',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
parser.add_argument('--model', default='mixer_b16_224', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                    help='ratio of validation batch size to training batch size (default: 1)')

# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')


# Learning rate schedule parameters
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='const',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--log-wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')

args, unknown = parser.parse_known_args()
wandb.init(config=args)
wandb.config.update(args)


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

###################################################################

def train_one_epoch(ADMM, 
                    epoch, model, loader, optimizer, loss_fn, args,
                    lr_scheduler=None, saver=None, amp_autocast=suppress):   

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.masked_retrain and not args.combine_progressive:
        print("!!!! Full acc re-train masking")
        masks = {}
        for name, W in (model.named_parameters()):
            weight = W.cpu().detach().numpy()
            non_zeros = weight != 0
            non_zeros = non_zeros.astype(np.float32)
            zero_mask = torch.from_numpy(non_zeros).cuda()
            W = torch.from_numpy(weight).cuda()
            W.data = W
            masks[name] = zero_mask
    elif args.combine_progressive:
        print("!!!! Progressive admm-train/re-train masking")
        masks = {}
        for name, W in (model.named_parameters()):
            weight = W.cpu().detach().numpy()
            non_zeros = weight != 0
            non_zeros = non_zeros.astype(np.float32)
            zero_mask = torch.from_numpy(non_zeros).cuda()
            W = torch.from_numpy(weight).cuda()
            W.data = W
            masks[name] = zero_mask

    model.train()

    total_ce = 0
    ctr = 0 

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)

    for batch_idx, (input, target) in enumerate(loader):
        
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)

        ctr += 1
        mixed_loss_sum = []
        loss = []

        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            output = model(input)
            ce_loss = loss_fn(output, target)
            total_ce = total_ce + float(ce_loss.item())

        if args.admm_train:
            # print("!!!! ADMM Train !!!!")
            admm.z_u_update(args, ADMM, model, 'cuda', loader, optimizer, epoch, input, batch_idx, None)  # update Z and U variables
            ce_loss, admm_loss, mixed_loss = admm.append_admm_loss(args, ADMM, model, ce_loss)  # append admm losss

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses_m.update(ce_loss.item(), input.size(0))
        
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # -------
        wandb.log({"iter/train_loss": ce_loss.item()})
        wandb.log({"iter/train_acc@1": acc1.item()})
        wandb.log({"iter/train_acc@5": acc5.item()})
        # -------

        optimizer.zero_grad()

        if args.admm_train:
            # mixed_loss.backward(create_graph=second_order)
            mixed_loss.backward(retain_graph=True, create_graph=second_order)
        else:
            ce_loss.backward()

        if args.clip_grad is not None:
            dispatch_clip_grad(
                model_parameters(model, exclude_head='agc' in args.clip_mode),
                value=args.clip_grad, mode=args.clip_mode)
        
        if args.combine_progressive:
            with torch.no_grad():
                for name, W in (model.named_parameters()):
                    if name in masks:
                        W.grad *= masks[name]
        if args.masked_retrain:
            with torch.no_grad():
                for name, W in (model.named_parameters()):
                    if name in masks:
                        W.grad *= masks[name]
        
        optimizer.step()
        
        torch.cuda.synchronize()

        # mixed_loss.backward(retain_graph=True)
        # for p in list(model.parameters()):
        #     if hasattr(p,'org'):
        #         p.data.copy_(p.org)
        # optimizer.step()
        # for p in list(model.parameters()):
        #         if hasattr(p,'org'):
        #             p.org.copy_(p.data.clamp_(-1.3,1.3))

        if args.admm_train:
            mixed_loss_sum.append(float(mixed_loss))

        loss.append(float(ce_loss))

        ### measure elapsed time
        num_updates += 1
        batch_time_m.update(time.time() - end)

        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            wandb.log({"Hyper/lr": lr})

        if batch_idx % args.print_freq == 0:

            _logger.info(
                'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                'LR: {lr:.3e}  '
                'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                    epoch, batch_idx, len(loader),
                    100. * batch_idx / last_idx,
                    loss=losses_m, batch_time=batch_time_m,
                    top1=top1, top5=top5,
                    lr=lr, data_time=data_time_m))
            _logger.info("cross_entropy loss: {}".format(ce_loss))

        # if saver is not None and args.recovery_interval and (
        #         last_batch or (batch_idx + 1) % args.recovery_interval == 0):
        #     saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)
        
        end = time.time()
        ### end for

    if args.admm_train:
        lossadmm = []
        for k, v in admm_loss.items():
            print("at layer {}, admm loss is {}".format(k, v))
            lossadmm.append(float(v))
        
    # if args.verbose:
    #     writer.add_scalar('Train/Cross_Entropy', ce_loss, epoch)
    #     for k, v in admm_loss.items():
    #         print("at layer {}, admm loss is {}".format(k, v))
    #         ADMM.admmloss[k].extend([float(v)])

    #     for k in ADMM.prune_ratios:
    #         writer.add_scalar('layer:{} Train/ADMM_Loss'.format(k), admm_loss[k], epoch)
            
    ADMM.ce_prev = ADMM.ce
    ADMM.ce = total_ce / ctr
    
    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg), 
                        ('top1', top1.avg), 
                        ('top5', top5.avg)])


# -----------------------------------------------
def validate(model, loader, loss_fn, args, amp_autocast=suppress):
    
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    end = time.time()
    last_idx = len(loader) - 1
    
    model.eval()

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):       
            last_batch = batch_idx == last_idx
            
            input = input.cuda()
            target = target.cuda()

            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            test_loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            reduced_loss = test_loss.data
        
            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            # -------
            wandb.log({"iter/test_loss": reduced_loss.item()})
            wandb.log({"iter/test_acc@1": acc1})
            wandb.log({"iter/test_acc@5": acc5})
            # -------

            ### measure elapsed time
            batch_time_m.update(time.time() - end)
            end = time.time()

            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                # print('Epoch: [{0}/{1}]\t'
                #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                #       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                #       'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                #             batch_idx, last_idx,
                #             batch_time=batch_time_m, loss=losses_m, 
                #             top1=top1_m, top5=top5_m))

                # --------------------
                log_name = 'Test'
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, 
                        batch_time=batch_time_m, loss=losses_m, 
                        top1=top1_m, top5=top5_m))
                # --------------------

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics


############################################
############################################

def main():

    setup_default_logging()
    args, args_text = _parse_args()
    
    if args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else: 
            warnings.warn("You've requested to log metrics to wandb but package not found. "
                            "Metrics not being logged to wandb, try `pip install wandb`")
    
    args.prefetcher = not args.no_prefetcher
    args.distributed = False

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    random_seed(args.seed)#, args.rank)


    ### set up model archetecture
    model = create_model(args.model,
                         pretrained=args.pretrained,
                         num_classes=args.num_classes,
                         in_chans=3,
                         global_pool=args.gp,
                         scriptable=args.torchscript)

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly
    
    _logger.info(
        f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    wandb.watch(model)

    # -------------------
    ### Load dataset
    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    ### setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    ### enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    ### move model to GPU, enable channels last layout if set
    # model = torch.nn.DataParallel(model).cuda()
    # model.cuda()

    # if args.apex_amp:
    #     model = amp.initialize(model, opt_level='O1')
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    
    if args.num_gpu > 1:
        # model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    # # -----------
    # print("||||||||||| Model Stru |||||||||||")
    # for i, (name, W) in enumerate(model.named_parameters()):
    #     if ("weight" in name) and (len(W.shape)>1):
    #         print(name)

    # for i, (name, W) in enumerate(model.named_parameters()):
    #     print(i, "th weight:", name, ", shape = ", W.shape, ", weight.dtype = ", W.dtype)
    # print("||||||||||| Model Stru |||||||||||")
    # # -------------



    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    ### setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            print('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            print('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            print('AMP not enabled. Training in float32.')


    ### optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
                            model, args.resume,
                            optimizer=None if args.no_resume_opt else optimizer,
                            loss_scaler=None if args.no_resume_opt else loss_scaler,
                            log_info=args.local_rank == 0)

    ### setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        ### a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    ### create the train and eval datasets
    dataset_train = create_dataset(args.dataset,
                                    root=args.data_dir, 
                                    split=args.train_split, 
                                    is_training=True,
                                    batch_size=args.batch_size, 
                                    repeats=args.epoch_repeats)
    dataset_eval = create_dataset(args.dataset, 
                                    root=args.data_dir, 
                                    split=args.val_split, 
                                    is_training=False, 
                                    batch_size=args.batch_size)

    ### setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    ### wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)


    ### create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader
    )

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size_multiplier * args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )

   ### setup loss function
    if args.jsd:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing).cuda()
    elif mixup_active:
        # smoothing is handled with mixup target transform
        train_loss_fn = SoftTargetCrossEntropy().cuda()
    elif args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
    else:
        train_loss_fn = nn.CrossEntropyLoss().cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()


    """=============="""
    """  ADMM Train  """
    """=============="""

    if args.admm_train:
        print("\n!!!!!!!!!!!!!!!!!!! ADMM TRAIN !!!!!!!!!!!!!!!!!\n")

        best_acc1 = 0

        ### setup checkpoint saver and eval metric tracking
        eval_metric = args.eval_metric
        model_ema = None
        best_metric = None
        best_epoch = None
        saver = None
        output_dir = None

        output_dir = '/data/shanglin/admm_train/timm_model'
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(model=model, optimizer=optimizer, 
                                args=args, model_ema=model_ema, 
                                amp_scaler=loss_scaler,
                                checkpoint_dir=output_dir, 
                                recovery_dir=output_dir, 
                                decreasing=decreasing, 
                                max_history=args.checkpoint_hist)

        initial_rho = args.rho

        for i in range(args.rho_num):
            
            current_rho = initial_rho * 10 ** i
            # if i == 0:
            #     print("-------> Initial model accuracy:")
            #     eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)
            #     print(eval_metrics)
            #     wandb.log({"test_acc@1": eval_metrics['top1']})
            #     wandb.log({"test_acc@5": eval_metrics['top5']})
            # else:
            #     model.load_state_dict(torch.load("model_prunned/imagenet_{}_{}_{}.pt".format(current_rho/10, args.config_file, args.sparsity_type)))
            #     model.cuda()

            ### check; Todo: modify config file
            ADMM = admm.ADMM(args, model, "./mlp_mixer_slr/profile/" + args.config_file + ".yaml", rho=current_rho)
            admm.admm_initialization(args, ADMM, model)  # intialize Z and U variables

            for epoch in range(1, args.epochs + 1):
                print("Epoch ", epoch)

                ### !!!!!! (change saver); Todo: modify training
                train_metrics = train_one_epoch(ADMM, 
                                                epoch, model, loader_train, 
                                                optimizer, train_loss_fn, args,
                                                lr_scheduler=lr_scheduler, 
                                                saver=None,
                                                amp_autocast=amp_autocast)
                
                wandb.log({"train_acc@1": train_metrics['top1']})
                wandb.log({"train_acc@5": train_metrics['top5']})
                
                ### check; Todo: modify validation
                eval_metrics = validate(model, 
                                        loader_eval, 
                                        validate_loss_fn, 
                                        args, 
                                        amp_autocast=amp_autocast)
                
                wandb.log({"test_acc@1": eval_metrics['top1']})
                wandb.log({"test_acc@5": eval_metrics['top5']})

                ### step LR for next epoch
                if lr_scheduler is not None:
                    lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

                ### todo: modify, whether needed
                if output_dir is not None:
                    summary_file = 'summary_{}{}.csv'.format(args.config_file, args.ext)
                    update_summary(epoch, train_metrics, eval_metrics, 
                                    os.path.join(output_dir, summary_file),
                                    write_header=best_metric is None, 
                                    log_wandb=args.log_wandb and has_wandb)
                
                
                if (best_acc1 < eval_metrics["top1"]) and (epoch != 1):
                    
                    ## remove old model
                    old_file = "mlpmix_imagenet_{}_{}_{}{}.pt".format(best_acc1, args.config_file, args.sparsity_type, args.ext)
                    if os.path.exists("/data/shanglin/admm_train/"+old_file):
                        os.remove("/data/shanglin/admm_train/"+old_file)

                    ### save new one
                    best_acc1 = max(eval_metrics["top1"], best_acc1)
                    model_best = model
                    torch.save(model_best.state_dict(), 
                                "/data/shanglin/admm_train/mlpmix_imagenet_{}_{}_{}{}.pt".format(best_acc1, 
                                                                                                 args.config_file, 
                                                                                                 args.sparsity_type,
                                                                                                 args.ext))
                ### save proper checkpoint with eval metric
                ### check; Todo: modify save checkpoint
                if (saver is not None) and (epoch != 1):
                    save_metric = eval_metrics[eval_metric]
                    best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)


                ### middle check
                if epoch%5 == 0: 
                    model_check = model
                    admm.hard_prune(args, ADMM, model_check)
                    compression = admm.test_sparsity(args, ADMM, model_check)

                    print("(middle check) accuracy after hard-pruning")
                    eval_metrics_mid = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)
                
                    wandb.log({"middle/test_acc@1": eval_metrics_mid['top1']})
                    wandb.log({"middle/test_acc@5": eval_metrics_mid['top5']})

                    print("(middle check) ACC: ", eval_metrics_mid['top1'], eval_metrics_mid['top5'])
                    # if eval_metrics_mid['top5'] > 86:
                    #     break


                print("Condition 1")
                print(ADMM.condition1)
                print("Condition 2")
                print(ADMM.condition2)

            dir_save = "mlp_mixer_slr/admm_model/admm_train/"
            torch.save(model_best.state_dict(), 
                        dir_save+"mlpmix_imagenet_{}_{}_{}{}.pt".format(best_acc1, 
                                                                        args.config_file, 
                                                                        args.sparsity_type,
                                                                        args.ext))

            if best_metric is not None:
                _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


            print("----------------> Accuracy after hard-pruning ...")
            model_forhard = model_best
            admm.hard_prune(args, ADMM, model_forhard)
            admm.test_sparsity(args, ADMM, model_forhard)

            ### check; Todo: modify validation
            eval_metrics_hp = validate(model_best, loader_eval, 
                                        validate_loss_fn, args, 
                                        amp_autocast=amp_autocast)

            # ### save proper checkpoint with eval metric
            # if saver is not None:
            #     save_metric = eval_metrics_hp[eval_metric]
            #     best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

            ### Todo: save hardprune model
            torch.save(model_forhard.state_dict(), 
                       "mlp_mixer_slr/admm_model/hardprune/mlpmix_imagenet_{}_{}_{}{}.pt".format(eval_metrics_hp['top1'], 
                                                                                                 args.config_file, 
                                                                                                 args.sparsity_type,
                                                                                                 args.ext))
    
    """================"""
    """End ADMM retrain"""
    """================"""

    """================"""
    """ Masked retrain """
    """================"""
    
    if args.masked_retrain:
        
        print("\n!!!!!!!!!!!!!!!!!!! RETRAIN !!!!!!!!!!!!!!!!!")

        ### setup checkpoint saver and eval metric tracking
        eval_metric = args.eval_metric
        model_ema = None
        best_metric = None
        best_epoch = None
        saver = None
        output_dir = None

        output_dir = '/data/shanglin/admm_retrain/timm_retrain'
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(model=model, optimizer=optimizer, 
                                args=args, model_ema=model_ema, 
                                amp_scaler=loss_scaler,
                                checkpoint_dir=output_dir, 
                                recovery_dir=output_dir, 
                                decreasing=decreasing, 
                                max_history=args.checkpoint_hist)

        print("\n---------------> Loading admm trained file...")
        filename_slr = "mlp_mixer_slr/admm_model/admm_train/mlpmix_imagenet_{}_{}_{}{}.pt".format(args.admmtrain_acc, 
                                                                                                    args.config_file, 
                                                                                                    args.sparsity_type,
                                                                                                    args.ext)
        print("!!! Loaded File: ", filename_slr)
        
        ### todo: modify
        load_checkpoint(model, filename_slr)
        model.cuda()

        ### check; Todo: modify validation
        print("\n---------------> Accuracy before hardpruning")
        eval_metrics_slr = validate(model, loader_eval, 
                                    validate_loss_fn, args, 
                                    amp_autocast=amp_autocast)

        wandb.log({"retrain_test_acc@1": eval_metrics_slr["top1"]})
        wandb.log({"retrain_test_acc@5": eval_metrics_slr["top5"]})

        best_acc1 = eval_metrics_slr["top1"]

        print("\n---------------> Accuracy after hard-pruning")
        ADMM = admm.ADMM(args, model, "./mlp_mixer_slr/profile/" + args.config_file + ".yaml", rho=args.rho)

        admm.hard_prune(args, ADMM, model)

        ### check; Todo: modify validation
        eval_metrics_hp = validate(model, loader_eval, 
                                    validate_loss_fn, args, 
                                    amp_autocast=amp_autocast)

        wandb.log({"retrain_test_acc@1": eval_metrics_hp["top1"]})
        wandb.log({"retrain_test_acc@5": eval_metrics_hp["top5"]})

        admm.test_sparsity(args, ADMM, model)


        for epoch in range(1, args.retrain_epoch+1):

            ### check; Todo: modify training
            retrain_metrics = train_one_epoch(ADMM, 
                                              epoch, model, loader_train, 
                                              optimizer, train_loss_fn, args,
                                              lr_scheduler=lr_scheduler, 
                                              saver=saver, 
                                              amp_autocast=amp_autocast)
                                 
            wandb.log({"retrain_train_acc@1": retrain_metrics["top1"]})
            wandb.log({"retrain_train_acc@5": retrain_metrics["top5"]})

            ### check; Todo: modify validation
            eval_metrics_rt = validate(model, loader_eval, 
                                        validate_loss_fn, args, 
                                        amp_autocast=amp_autocast)

            wandb.log({"retrain_test_acc@1": eval_metrics_rt["top1"]})
            wandb.log({"retrain_test_acc@5": eval_metrics_rt["top5"]})
            

            ### step LR for next epoch
            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1, eval_metrics_rt[eval_metric])
            
            ### todo: modify, whether needed
            if output_dir is not None:
                summary_file = 'summary_retrain_'+args.config_file+'.csv'
                update_summary(epoch, retrain_metrics, eval_metrics_rt, 
                                os.path.join(output_dir, summary_file),
                                write_header=best_metric is None, 
                                log_wandb=args.log_wandb and has_wandb)

            if best_acc1 < eval_metrics_rt["top1"]:

                ## remove old model
                old_file_rt = "mlpmix_imagenet_{}_{}_{}{}.pt".format(best_acc1, args.config_file, args.sparsity_type, args.ext)
                if os.path.exists("/data/shanglin/admm_retrain/"+old_file_rt):
                    os.remove("/data/shanglin/admm_retrain/"+old_file_rt)

                ### save new one
                best_acc1 = max(eval_metrics_rt["top1"], best_acc1)
                model_best_retrain = model
                print("\n>_ Got better accuracy, saving model with top1 accuracy {:.3f}% now...\n".format(best_acc1))
                torch.save(model_best_retrain.state_dict(), 
                            "/data/shanglin/admm_retrain/mlpmix_imagenet_retrained_acc_{:.3f}_{}_{}{}.pt".format(best_acc1, 
                                                                                                                 args.config_file, 
                                                                                                                 args.sparsity_type,
                                                                                                                 args.ext))
            ### check; Todo: modify save checkpoint
            if saver is not None:
                save_metric = eval_metrics_rt[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)


        dir_save = "mlp_mixer_slr/admm_model/retrain/"
        torch.save(model_best_retrain.state_dict(), 
                    dir_save+"mlpmix_imagenet_{}_{}_{}{}.pt".format(best_acc1, 
                                                                    args.config_file, 
                                                                    args.sparsity_type,
                                                                    args.ext))
        print("---------------> After retraining")
        eval_metrics_final = validate(model_best_retrain, loader_eval, 
                                        validate_loss_fn, args, 
                                        amp_autocast=amp_autocast)
        admm.test_sparsity(args, ADMM, model_best_retrain)


    """=================="""
    """End masked retrain"""
    """=================="""

            
if __name__ == '__main__':
    main()