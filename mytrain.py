import sys
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import warnings
import torch.distributed
import numpy as np
import random
import faulthandler
import torch.multiprocessing as mp
import time
import scipy.misc
from models.networks import PointFlow
from torch import optim
from args import get_args
from torch.backends import cudnn
from utils import AverageValueMeter, set_random_seed, apply_random_rotation, save, resume, visualize_point_clouds
from tensorboardX import SummaryWriter
from datasets import get_datasets, init_np_seed, MyDataset

faulthandler.enable()


def main_worker(gpu, save_dir, ngpus_per_node, args):
    # basic setup
    cudnn.benchmark = True
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.log_name is not None:
        log_dir = "runs/%s" % args.log_name
    else:
        log_dir = "runs/time-%d" % time.time()

    if not args.distributed or (args.rank % ngpus_per_node == 0):
        writer = SummaryWriter(logdir=log_dir)
    else:
        writer = None

    if not args.use_latent_flow:  # auto-encoder only
        args.prior_weight = 0
        args.entropy_weight = 0

    # multi-GPU setup
    model = PointFlow(args)
    if args.distributed:  # Multiple processes, single GPU per process
        if args.gpu is not None:
            def _transform_(m):
                return nn.parallel.DistributedDataParallel(
                    m, device_ids=[args.gpu], output_device=args.gpu, check_reduction=True)

            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model.multi_gpu_wrapper(_transform_)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = 0
        else:
            assert 0, "DistributedDataParallel constructor should always set the single device scope"
    elif args.gpu is not None:  # Single process, single GPU per process
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:  # Single process, multiple GPUs per process
        def _transform_(m):
            return nn.DataParallel(m)
        model = model.cuda()
        model.multi_gpu_wrapper(_transform_)

    # resume checkpoints
    start_epoch = 0
    optimizer = model.make_optimizer(args)
    if args.resume_checkpoint is None and os.path.exists(os.path.join(save_dir, 'checkpoint-latest.pt')):
        args.resume_checkpoint = os.path.join(save_dir, 'checkpoint-latest.pt')  # use the latest checkpoint
    if args.resume_checkpoint is not None:
        if args.resume_optimizer:
            model, optimizer, start_epoch = resume(
                args.resume_checkpoint, model, optimizer, strict=(not args.resume_non_strict))
        else:
            model, _, start_epoch = resume(
                args.resume_checkpoint, model, optimizer=None, strict=(not args.resume_non_strict))
        print('Resumed from: ' + args.resume_checkpoint)

    # initialize datasets and loaders
    tr_dataset = MyDataset(args.data_dir, istest = False) 
    te_dataset = MyDataset(args.data_dir, istest=True) 
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(tr_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        dataset=tr_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=0, pin_memory=True, sampler=train_sampler, drop_last=True,
        worker_init_fn=init_np_seed)
    test_loader = torch.utils.data.DataLoader(
        dataset=te_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
        worker_init_fn=init_np_seed)

    # save dataset statistics
    # if not args.distributed or (args.rank % ngpus_per_node == 0):
    #     np.save(os.path.join(save_dir, "train_set_mean.npy"), tr_dataset.all_points_mean)
    #     np.save(os.path.join(save_dir, "train_set_std.npy"), tr_dataset.all_points_std)
    #     np.save(os.path.join(save_dir, "train_set_idx.npy"), np.array(tr_dataset.shuffle_idx))
    #     np.save(os.path.join(save_dir, "val_set_mean.npy"), te_dataset.all_points_mean)
    #     np.save(os.path.join(save_dir, "val_set_std.npy"), te_dataset.all_points_std)
    #     np.save(os.path.join(save_dir, "val_set_idx.npy"), np.array(te_dataset.shuffle_idx))

    # load classification dataset if needed
    if args.eval_classification:
        from datasets import get_clf_datasets

        def _make_data_loader_(dataset):
            return torch.utils.data.DataLoader(
                dataset=dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=0, pin_memory=True, drop_last=False,
                worker_init_fn=init_np_seed
            )

        clf_datasets = get_clf_datasets(args)
        clf_loaders = {
            k: [_make_data_loader_(ds) for ds in ds_lst] for k, ds_lst in clf_datasets.items()
        }
    else:
        clf_loaders = None

    # initialize the learning rate scheduler
    if args.scheduler == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.exp_decay)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 2, gamma=0.1)
    elif args.scheduler == 'linear':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - 0.5 * args.epochs) / float(0.5 * args.epochs)
            return lr_l
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    else:
        assert 0, "args.schedulers should be either 'exponential' or 'linear'"

    # main training loop
    start_time = time.time()
    entropy_avg_meter = AverageValueMeter()
    latent_nats_avg_meter = AverageValueMeter()
    point_nats_avg_meter = AverageValueMeter()
    if args.distributed:
        print("[Rank %d] World size : %d" % (args.rank, dist.get_world_size()))

    print("Start epoch: %d End epoch: %d" % (start_epoch, args.epochs))
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # adjust the learning rate
        if (epoch + 1) % args.exp_decay_freq == 0:
            scheduler.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar('lr/optimizer', scheduler.get_lr()[0], epoch)

        # train for one epoch
        for bidx, data in enumerate(train_loader):
            idx_batch, tr_batch, te_batch = data['idx'], data['train_points'], data['test_points']
            step = bidx + len(train_loader) * epoch
            model.train()
            inputs = tr_batch.cuda(args.gpu, non_blocking=True)
            out = model(inputs, optimizer, step, writer)
            entropy, prior_nats, recon_nats = out['entropy'], out['prior_nats'], out['recon_nats']
            entropy_avg_meter.update(entropy)
            point_nats_avg_meter.update(recon_nats)
            latent_nats_avg_meter.update(prior_nats)
            if step % args.log_freq == 0:
                duration = time.time() - start_time
                start_time = time.time()
                print("[Rank %d] Epoch %d Batch [%2d/%2d] Time [%3.2fs] Entropy %2.5f LatentNats %2.5f PointNats %2.5f"
                      % (args.rank, epoch, bidx, len(train_loader), duration, entropy_avg_meter.avg,
                         latent_nats_avg_meter.avg, point_nats_avg_meter.avg))

        # evaluate on the validation set
        # if not args.no_validation and (epoch + 1) % args.val_freq == 0:
        #     from utils import validate
        #     validate(test_loader, model, epoch, writer, save_dir, args, clf_loaders=clf_loaders)

        # save visualizations
        if (epoch + 1) % args.viz_freq == 0:
            # reconstructions
            model.eval()
            samples = model.reconstruct(inputs)
            results = []
            for idx in range(min(10, inputs.size(0))):
                res = visualize_point_clouds(samples[idx], inputs[idx], idx)
                results.append(res)
            res = np.concatenate(results, axis=1)
            scipy.misc.imsave(os.path.join(save_dir, 'images', 'tr_vis_conditioned_epoch%d-gpu%s.png' % (epoch, args.gpu)),
                              res.transpose((1, 2, 0)))
            if writer is not None:
                writer.add_image('tr_vis/conditioned', torch.as_tensor(res), epoch)

            # samples
            if args.use_latent_flow:
                num_samples = min(10, inputs.size(0))
                num_points = inputs.size(1)
                _, samples = model.sample(num_samples, num_points)
                results = []
                for idx in range(num_samples):
                    res = visualize_point_clouds(samples[idx], inputs[idx], idx)
                    results.append(res)
                res = np.concatenate(results, axis=1)
                scipy.misc.imsave(os.path.join(save_dir, 'images', 'tr_vis_conditioned_epoch%d-gpu%s.png' % (epoch, args.gpu)),
                                  res.transpose((1, 2, 0)))
                if writer is not None:
                    writer.add_image('tr_vis/sampled', torch.as_tensor(res), epoch)

        # save checkpoints
        if not args.distributed or (args.rank % ngpus_per_node == 0):
            if (epoch + 1) % args.save_freq == 0:
                save(model, optimizer, epoch + 1,
                     os.path.join(save_dir, 'checkpoint-%d.pt' % epoch))
                save(model, optimizer, epoch + 1,
                     os.path.join(save_dir, 'checkpoint-latest.pt'))


def main():
    # command line args
    args = get_args()
    save_dir = os.path.join("checkpoints", args.log_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, 'images'))

    with open(os.path.join(save_dir, 'command.sh'), 'w') as f:
        f.write('python -X faulthandler ' + ' '.join(sys.argv))
        f.write('\n')

    if args.seed is None:
        args.seed = random.randint(0, 1000000)
    set_random_seed(args.seed)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    if args.sync_bn:
        assert args.distributed

    print("Arguments:")
    print(args)

    ngpus_per_node = torch.cuda.device_count()
    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(save_dir, ngpus_per_node, args))
    else:
        main_worker(args.gpu, save_dir, ngpus_per_node, args)


if __name__ == '__main__':
    main()
