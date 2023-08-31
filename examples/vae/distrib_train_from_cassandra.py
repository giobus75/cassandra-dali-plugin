# Adapted to use cassandra-dali-plugin, from
# https://github.com/NVIDIA/DALI/blob/main/docs/examples/use_cases/pytorch/resnet50/main.py
# (Apache License, Version 2.0)

# cassandra reader
from cassandra_reader import get_cassandra_reader
from crs4.cassandra_utils import get_shard

import argparse
import os
import shutil
import time
import math
import pickle

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

#import torchvision.models as models
import models 

import numpy as np

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except ImportError:
    raise ImportError(
        "Please install DALI from https://www.github.com/NVIDIA/DALI to run this example."
    )


# supporting torchrun
global_rank = int(os.getenv("RANK", default=0))
local_rank = int(os.getenv("LOCAL_RANK", default=0))
world_size = int(os.getenv("WORLD_SIZE", default=1))


def parse():
    model_names = [name for name in models.vae_models]

    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    parser.add_argument(
        "--split-fn",
        metavar="FILENAME",
        required=True,
        help="split file filename",
    )
    parser.add_argument(
        "--train-index",
        metavar="TINDEX",
        default=0,
        type=int,
        help="Index of the split array in the splitfile to be used for training",
    )
    parser.add_argument(
        "--val-index",
        metavar="VINDEX",
        default=1,
        type=int,
        help="Index of the split array in the splitfile to be used for validation",
    )
    parser.add_argument(
        "--crossval-index",
        metavar="CVINDEX",
        default=None,
        type=int,
        help="Index of the split array in the splitfile to be used for validation.\
                The remaining splits, except the one specified with the --exclude-split option,\
                are merged and used as training data.\
                The --train-index and --val-index options will be overridden",
    )
    parser.add_argument(
        "--exclude-index",
        metavar="EINDEX",
        default=None,
        type=int,
        help="Index of the split to be excluded during the crossvalidation process.",
    )
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="VanillaVAE",
        choices=model_names,
        help="model architecture: " + " | ".join(model_names) + " (default: resnet18)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size per process (default: 256)",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.005,
        type=float,
        metavar="LR",
        help="Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.",
    )
    parser.add_argument(
        "--weight-decay",
        "--wd",
        default=0.0,
        type=float,
        metavar="W",
        help="weight decay (default: 0.0)",
    )
    parser.add_argument(
        "--print-freq",
        "-p",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )

    parser.add_argument(
        "--dali_cpu",
        action="store_true",
        help="Runs CPU based version of DALI pipeline.",
    )
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--sync_bn", action="store_true", help="enabling apex sync BN.")

    parser.add_argument("--opt-level", type=str, default=None)
    parser.add_argument("--keep-batchnorm-fp32", type=str, default=None)
    parser.add_argument("--loss-scale", type=str, default=None)
    parser.add_argument("--channels-last", type=bool, default=False)
    args = parser.parse_args()
    return args


# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, "item"):
        return t.item()
    else:
        return t[0]


@pipeline_def
def create_dali_pipeline(
    keyspace,
    table_suffix,
    id_col,
    label_type,
    data_col,
    bs,
    crop,
    size,
    dali_cpu=False,
    is_training=True,
    prefetch_buffers=8,
    io_threads=1,
    comm_threads=2,
    copy_threads=2,
    wait_threads=2,
):
    cass_reader = get_cassandra_reader(
        keyspace=keyspace,
        table_suffix=table_suffix,
        id_col=id_col,
        label_type=label_type,
        data_col=data_col,
        batch_size=bs,
        prefetch_buffers=prefetch_buffers,
        io_threads=io_threads,
        comm_threads=comm_threads,
        copy_threads=copy_threads,
        wait_threads=wait_threads,
    )
    images, labels = cass_reader
    dali_device = "cpu" if dali_cpu else "gpu"
    decoder_device = "cpu" if dali_cpu else "mixed"
    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 211025920 if decoder_device == "mixed" else 0
    host_memory_padding = 140544512 if decoder_device == "mixed" else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == "mixed" else 0
    preallocate_height_hint = 6430 if decoder_device == "mixed" else 0
    if is_training:
        images = fn.decoders.image_random_crop(
            images,
            device=decoder_device,
            output_type=types.RGB,
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
            preallocate_width_hint=preallocate_width_hint,
            preallocate_height_hint=preallocate_height_hint,
            random_aspect_ratio=[0.8, 1.25],
            random_area=[0.1, 1.0],
            num_attempts=100,
        )
        images = fn.resize(
            images,
            device=dali_device,
            resize_x=crop,
            resize_y=crop,
            interp_type=types.INTERP_TRIANGULAR,
        )
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images, device=decoder_device, output_type=types.RGB)
        images = fn.resize(
            images,
            device=dali_device,
            size=size,
            mode="not_smaller",
            interp_type=types.INTERP_TRIANGULAR,
        )
        mirror = False

    images = fn.crop_mirror_normalize(
        images.gpu(),
        dtype=types.FLOAT,
        output_layout="CHW",
        crop=(crop, crop),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=mirror,
    )
    labels = labels.gpu()
    return (images, labels)


def read_split_file(split_fn):
    data = pickle.load(open(split_fn, "rb"))
    keyspace = data["keyspace"]
    table_suffix = data["table_suffix"]
    id_col = data["id_col"]
    data_col = data["data_col"]  # Name of the table column with actual data
    label_type = data["label_type"]
    row_keys = data["row_keys"]  # Numpy array of UUIDs
    split = data[
        "split"
    ]  # List of arrays. Each arrays indexes the row_keys array for each split.

    return (
        keyspace,
        table_suffix,
        id_col,
        data_col,
        label_type,
        row_keys,
        split,
    )


def compute_split_index(split, train_index, val_index, crossval_index, exclude_index):
    n_split = len(split)

    # Merge splits for training samples if crossvalidation is requested.
    # Do nothing otherwise
    if crossval_index and n_split > 2:
        if exclude_index > n_split:
            exlcude_index = n_split - 1
        if crossval_index > n_split or crossval_index == exclude_index:
            crossval_index = n_split - 2

        tis = np.array(
            [i for i in range(n_split) if i != exclude_index and i != val_index]
        )
        train_split = np.concatenate([split[i] for i in tis])
        val_split = split[val_index]

        split = [train_split, val_split]
        train_index = 0
        val_index = 1

        print("\nCrossvalidation:")
        print(f"Training samples will be taken from splits {tis}")
        print(f"Validation samples will be taken from split {crossval_index}")
        if exclude_index:
            print(f"Split {exclude_index} will not be used")
        print("\n")

    return split, train_index, val_index


def main():
    global best_loss, args
    best_loss = np.inf
    args = parse()

    ## Read split file to get data for training
    (
        keyspace,
        table_suffix,
        id_col,
        data_col,
        label_type,
        row_keys,
        split,
    ) = read_split_file(args.split_fn)

    # Get split indexes
    split, train_index, val_index = compute_split_index(
        split, args.train_index, args.val_index, args.crossval_index, args.exclude_index
    )

    args.distributed = world_size > 1

    # make apex optional
    if args.opt_level is not None or args.distributed or args.sync_bn:
        try:
            global DDP, amp, optimizers, parallel
            from apex.parallel import DistributedDataParallel as DDP
            from apex import amp, optimizers, parallel
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to run this example."
            )

    print("opt_level = {}".format(args.opt_level))
    print(
        "keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32),
        type(args.keep_batchnorm_fp32),
    )
    print("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))

    print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    cudnn.benchmark = True
    best_prec1 = 0
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(local_rank)  # global_rank ?
        torch.set_printoptions(precision=10)

    args.gpu = 0

    if args.distributed:
        args.gpu = local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    args.total_batch_size = world_size * args.batch_size
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    # create vanilla_vae model
    patch_size = 64
    model_config = {'in_channels': 3, 'latent_dim': 128, 'patch_size': patch_size}
    model = models.vae_models[args.arch](**model_config)
    criterion = model.loss_function

    if args.sync_bn:
        print("using apex synced BN")
        model = parallel.convert_syncbn_model(model)

    if hasattr(torch, "channels_last") and hasattr(torch, "contiguous_format"):
        if args.channels_last:
            memory_format = torch.channels_last
        else:
            memory_format = torch.contiguous_format
        model = model.cuda().to(memory_format=memory_format)
    else:
        model = model.cuda()

    # Scale learning rate based on global batch size
    optimizer = torch.optim.Adam(
        model.parameters(),
        args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)

    # Initialize Amp.  Amp accepts either values or strings for the
    # optional override arguments, for convenient interoperation with
    # argparse.
    if args.opt_level is not None:
        model, optimizer = amp.initialize(
            model,
            optimizer,
            opt_level=args.opt_level,
            keep_batchnorm_fp32=args.keep_batchnorm_fp32,
            loss_scale=args.loss_scale,
        )

    # For distributed training, wrap the model with
    # apex.parallel.DistributedDataParallel.  This must be done AFTER
    # the call to amp.initialize.  If model = DDP(model) is called
    # before model, ... = amp.initialize(model, ...), the call to
    # amp.initialize may alter the types of model's parameters in a
    # way that disrupts or destroys DDP's allreduce hooks.
    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps
        # communication with computation in the backward pass.  model
        # = DDP(model) delay_allreduce delays all communication to the
        # end of the backward pass.
        model = DDP(model, delay_allreduce=True)

    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(
                    args.resume,
                    map_location=lambda storage, loc: storage.cuda(args.gpu),
                )
                args.start_epoch = checkpoint["epoch"]
                global best_prec1
                best_prec1 = checkpoint["best_prec1"]
                model.load_state_dict(checkpoint["state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                print(
                    "=> loaded checkpoint '{}' (epoch {})".format(
                        args.resume, checkpoint["epoch"]
                    )
                )
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        resume()

    # train pipe
    train_uuids = row_keys[split[train_index]]

    pipe = create_dali_pipeline(
        keyspace=keyspace,
        table_suffix=table_suffix,
        id_col=id_col,
        label_type=label_type,
        data_col=data_col,
        batch_size=args.batch_size,
        bs=args.batch_size,
        num_threads=args.workers,
        device_id=local_rank,
        seed=12 + local_rank,  # global_rank?
        crop=patch_size,
        size=patch_size,
        dali_cpu=args.dali_cpu,
        is_training=True,
    )
    pipe.build()

    # pre-feeding train pipeline
    uuids, real_sz = get_shard(
        train_uuids,
        batch_size=args.batch_size,
        epoch=0,
        shard_id=local_rank,  # global_rank?
        num_shards=world_size,
    )
    for u in uuids:
        pipe.feed_input("Reader[0]", u)
    train_loader = DALIClassificationIterator(
        pipe, size=real_sz, last_batch_policy=LastBatchPolicy.PARTIAL
    )

    # val pipe
    val_uuids = row_keys[split[val_index]]

    pipe = create_dali_pipeline(
        keyspace=keyspace,
        table_suffix=table_suffix,
        id_col=id_col,
        label_type=label_type,
        data_col=data_col,
        batch_size=args.batch_size,
        bs=args.batch_size,
        num_threads=args.workers,
        device_id=local_rank,
        seed=12 + local_rank,  # global_rank?
        crop=patch_size,
        size=patch_size,
        dali_cpu=args.dali_cpu,
        is_training=False,
    )
    pipe.build()

    # pre-feeding val pipeline
    uuids, real_sz = get_shard(
        val_uuids,
        batch_size=args.batch_size,
        epoch=0,
        shard_id=local_rank,  # global_rank?
        num_shards=world_size,
    )
    for u in uuids:
        pipe.feed_input("Reader[0]", u)
    val_loader = DALIClassificationIterator(
        pipe, size=real_sz, last_batch_policy=LastBatchPolicy.PARTIAL
    )

    total_time = AverageMeter()
    for epoch in range(args.start_epoch, args.epochs):
        # pre-feeding train pipeline
        uuids, real_sz = get_shard(
            train_uuids,
            batch_size=args.batch_size,
            epoch=1 + epoch,
            shard_id=local_rank,  # global_rank?
            num_shards=world_size,
        )
        for u in uuids:
            train_loader._pipes[0].feed_input("Reader[0]", u)
        # pre-feeding val  pipeline
        uuids, real_sz = get_shard(
            val_uuids,
            batch_size=args.batch_size,
            epoch=1 + epoch,
            shard_id=local_rank,  # global_rank?
            num_shards=world_size,
        )
        for u in uuids:
            val_loader._pipes[0].feed_input("Reader[0]", u)

        # train for one epoch
        avg_train_time = train(train_loader, model, optimizer, criterion, epoch)
        total_time.update(avg_train_time)

        # evaluate on validation set
        val_loss = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        if local_rank == 0:  # global_rank?
            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
            )
            if epoch == args.epochs - 1:
                print(
                    "##Loss {0}\n".format(
                        val_loss
                    )
                )

        train_loader.reset()
        val_loader.reset()


def train(train_loader, model, optimizer, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    rec_losses = AverageMeter()
    kld_losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, data in enumerate(train_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze(-1).long()
        train_loader_len = int(math.ceil(train_loader._size / args.batch_size))

        # compute output
        output = model(input)
        loss = criterion(*output,
                                   M_N = 0.00025 #KL Divergence loss weight,
                                   )

        rec_loss = loss['Reconstruction_Loss']
        kld_loss = loss['KLD']
        loss = loss['loss']

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if args.opt_level is not None:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        if i % args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy,
            # and speed.  For best performance, it doesn't make sense
            # to print these metrics every iteration, since they incur
            # an allreduce and some host<->device syncs.

            # Average loss and accuracy across processes for logging
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
                reduced_rec_loss = reduce_tensor(rec_loss.data)
                reduced_kld_loss = reduce_tensor(kld_loss.data)
            else:
                reduced_loss = loss.data
                reduced_rec_loss = rec_loss.data
                reduced_kld_loss = kld_loss.data

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), input.size(0))
            rec_losses.update(to_python_float(reduced_rec_loss), input.size(0))
            kld_losses.update(to_python_float(reduced_kld_loss), input.size(0))
            torch.cuda.synchronize()
            
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            if local_rank == 0:  # global_rank?
                speed_val = world_size * args.batch_size / batch_time.val
                speed_avg = world_size * args.batch_size / batch_time.avg
                print(
                    f"Epoch: [{epoch}][{i}/{train_loader_len}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f}) - Speed {speed_val:.3f} ({speed_avg:.3f})\t \
                    Rec Loss {rec_losses.val:.4f} ({rec_losses.avg:.4f}) - KL Div {kld_losses.val:.4f} ({kld_losses.avg:.4f}) - Loss {losses.val:.4f} ({losses.avg:.4f})"
                    )
                

    return batch_time.avg


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, data in enumerate(val_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze(-1).long()
        val_loader_len = int(val_loader._size / args.batch_size)

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(*output,
                                   M_N = 1.0 #KL Divergence loss weight,
                                   )
        rec_loss = loss['Reconstruction_Loss']
        kld_loss = loss['KLD']
        loss = loss['loss']
        
        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO:  Change timings to mirror train().
        if local_rank == 0 and i % args.print_freq == 0:  # global_rank?
            print(
                "Test: [{0}/{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Speed {2:.3f} ({3:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})".format(
                    i,
                    val_loader_len,
                    world_size * args.batch_size / batch_time.val,
                    world_size * args.batch_size / batch_time.avg,
                    batch_time=batch_time,
                    loss=losses
                )
            )

    return loss


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt


if __name__ == "__main__":
    main()
