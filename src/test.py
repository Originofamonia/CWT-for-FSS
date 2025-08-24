import os
import random
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.optim as optim
from collections import defaultdict
import torch.distributed as dist
from tqdm import tqdm
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import time
from typing import Tuple

from src.dataset.dataset import get_val_loader
from src.util import AverageMeter, batch_intersectionAndUnionGPU, get_model_dir, get_model_dir_trans
from src.util import find_free_port, setup, cleanup, to_one_hot, intersectionAndUnionGPU
from src.model.pspnet import get_model_fs
from src.model.transformer import MultiHeadAttentionOne
from src.util import load_cfg_from_cfg_file, merge_cfg_from_list, mask_pooling_single_class


def parse_args() -> None:
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg


def main_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:

    print(f"==> Running DDP checkpoint example on rank {rank}.")
    setup(args, rank, world_size)

    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.manual_seed + rank)
        np.random.seed(args.manual_seed + rank)
        torch.manual_seed(args.manual_seed + rank)
        torch.cuda.manual_seed_all(args.manual_seed + rank)
        random.seed(args.manual_seed + rank)

    # ====== Model  ======
    model = get_model_fs(args).to(rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])

    trans_dim = args.bottleneck_dim
    transformer = MultiHeadAttentionOne(
        args.heads, trans_dim, trans_dim, trans_dim, dropout=0.5
    ).to(rank)

    transformer = nn.SyncBatchNorm.convert_sync_batchnorm(transformer)
    transformer = DDP(transformer, device_ids=[rank])
    
    root_trans = get_model_dir_trans(args)

    if args.resume_weights:
        if os.path.isfile(args.resume_weights):
            print("=> loading weight '{}'".format(args.resume_weights))

            pre_weight = torch.load(args.resume_weights)['state_dict']

            pre_dict = model.state_dict()
            for index, (key1, key2) in enumerate(zip(pre_dict.keys(), pre_weight.keys())):
                if 'classifier' not in key1 and index < len(pre_dict.keys()):
                    if pre_dict[key1].shape == pre_weight[key2].shape:
                        pre_dict[key1] = pre_weight[key2]
                    else:
                        print('Pre-trained {} shape and model {} shape: {}, {}'.
                              format(key2, key1, pre_weight[key2].shape, pre_dict[key1].shape))
                        continue

            model.load_state_dict(pre_dict, strict=True)

            print("=> loaded weight '{}'".format(args.resume_weights))
        else:
            print("=> no weight found at '{}'".format(args.resume_weights))

    if args.ckpt_used is not None:
        filepath = os.path.join(root_trans, f'{args.ckpt_used}.pth')
        assert os.path.isfile(filepath), filepath
        print("=> loading transformer weight '{}'".format(filepath))
        checkpoint = torch.load(filepath)
        transformer.load_state_dict(checkpoint['state_dict'])
        print("=> loaded transformer weight '{}'".format(filepath))
    else:
        print("=> Not loading anything")

    # ====== Data  ======
    episodic_val_loader, _ = get_val_loader(args)

    # ====== Test  ======
    val_Iou, val_loss = validate_transformer(
        args=args, val_loader=episodic_val_loader,
        model=model, transformer=transformer
    )

    if args.distributed:
        dist.all_reduce(val_Iou), dist.all_reduce(val_loss)
        val_Iou /= world_size
        val_loss /= world_size

    cleanup()


def validate_transformer(
        args: argparse.Namespace,
        val_loader: torch.utils.data.DataLoader,
        model: DDP,
        transformer: DDP
) -> Tuple[torch.tensor, torch.tensor]:

    print('==> Start testing')

    model.eval()
    transformer.eval()
    # n_episodes = int(args.test_num / args.batch_size_val)

    # ====== Metrics initialization  ======
    H, W = args.image_size, args.image_size
    # c = model.bottleneck_dim
    if args.image_size == 473:
        h = 60
        w = 60
    else:
        h = model.feature_res[0]
        w = model.feature_res[1]

    runtimes = torch.zeros(args.n_runs)
    val_IoUs = np.zeros(args.n_runs)
    val_losses = np.zeros(args.n_runs)

    # ====== Perform the runs  ======
    for run in range(args.n_runs):  # 1

        # ====== Initialize the metric dictionaries ======
        loss_meter = AverageMeter()
        iter_num = 0
        cls_intersection = defaultdict(int)  # Default value is 0
        cls_union = defaultdict(int)
        IoU = defaultdict(int)
        runtime = 0

        # for e in range(n_episodes):
        t0 = time.time()
        logits_q = torch.zeros(args.test_num, 1, args.num_classes_tr, h, w).to('cuda') # [100,1,2,60,60]
        gt_q = 255 * torch.ones(
            args.test_num, 1, args.image_size,args.image_size
        ).long().to('cuda')  # [100,1,473,473]
        classes = []  # All classes considered in the tasks

        # ====== Process each task separately ======
        # Batch size val is 50 here.

        pbar = tqdm(val_loader)
        for i, batch in enumerate(pbar):
            # try:
            qry_img, q_label, spprt_imgs, s_label, subcls, spprt_oris, qry_oris = batch
            # except:
            #     continue
            iter_num += 1

            spprt_imgs = spprt_imgs.to('cuda', non_blocking=True)  # [1,1,3,473,473]
            s_label = s_label.to('cuda', non_blocking=True)  # [1,1,473,473]

            q_label = q_label.to('cuda', non_blocking=True)  # [1,473,473]
            qry_img = qry_img.to('cuda', non_blocking=True)  # [1,3,473,473]

            # ====== Phase 1: Train a new binary classifier on support samples. ======
            binary_classifier = nn.Conv2d(
                args.bottleneck_dim, args.num_classes_tr, kernel_size=1, bias=False
            ).cuda()

            optimizer = optim.SGD(binary_classifier.parameters(), lr=args.cls_lr)

            # Dynamic class weights
            s_label_arr = s_label.cpu().numpy().copy()  # [n_task, n_shots, img_size, img_size]
            back_pix = np.where(s_label_arr == 0)
            target_pix = np.where(s_label_arr == 1)

            criterion = nn.CrossEntropyLoss(
                weight=torch.tensor([1.0, len(back_pix[0]) / len(target_pix[0])]).cuda(),
                ignore_index=255
            )

            with torch.no_grad():
                f_s = model.extract_features(spprt_imgs.squeeze(0))  # [n_task, n_shots, c, h, w]

            for index in range(args.adapt_iter):
                output_support = binary_classifier(f_s)
                output_support = F.interpolate(
                    output_support, size=s_label.size()[2:],
                    mode='bilinear', align_corners=True
                )
                s_loss = criterion(output_support, s_label.squeeze(0))
                optimizer.zero_grad()
                s_loss.backward()
                optimizer.step()

            # ====== Phase 2: Update classifier's weights with old weights and query features. ======
            with torch.no_grad():
                f_q = model.extract_features(qry_img)  # [n_task, c, h, w]
                f_q = F.normalize(f_q, dim=1)

                weights_cls = binary_classifier.weight.data  # [2, c, 1, 1]

                weights_cls_reshape = weights_cls.squeeze().unsqueeze(0).expand(
                    f_q.shape[0], 2, 512
                )  # [1, 2, c]

                updated_weights_cls = transformer(weights_cls_reshape, f_q, f_q)  # [1, 2, c]

                # Build a temporary new classifier for prediction
                Pseudo_cls = nn.Conv2d(
                    args.bottleneck_dim, args.num_classes_tr, kernel_size=1, bias=False
                ).cuda()

                # Initialize the weights with updated ones
                Pseudo_cls.weight.data = torch.as_tensor(
                    updated_weights_cls.squeeze(0).unsqueeze(2).unsqueeze(3)
                )

                pred_q = Pseudo_cls(f_q)

            logits_q[i] = pred_q.detach()
            gt_q[i, 0] = q_label
            classes.append([class_.item() for class_ in subcls])
            pbar.set_description(f'iter: {i}')

        t1 = time.time()
        runtime += t1 - t0

        logits = F.interpolate(
            logits_q.squeeze(1), size=(H, W),
            mode='bilinear', align_corners=True
        ).detach()  # logits: [100,2,473,473]

        intersection, union, _ = batch_intersectionAndUnionGPU(
            logits.unsqueeze(1), gt_q, 2
        )  # gt_q: [100,1,473,473]

        intersection, union = intersection.cpu(), union.cpu()

        # ====== Log metrics ======
        criterion_standard = nn.CrossEntropyLoss(ignore_index=255)
        loss = criterion_standard(logits, gt_q.squeeze(1))
        loss_meter.update(loss.item())
        for i, task_classes in enumerate(classes):
            for j, class_ in enumerate(task_classes):
                cls_intersection[class_] += intersection[i, 0, j + 1]  # Do not count background
                cls_union[class_] += union[i, 0, j + 1]

        for class_ in cls_union:
            IoU[class_] = cls_intersection[class_] / (cls_union[class_] + 1e-10)

        if (iter_num % 200 == 0):
            mIoU = np.mean([IoU[i] for i in IoU])
            print('Test: [{}/{}] '
                    'mIoU {:.4f} '
                    'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '.format(
                iter_num, args.test_num, mIoU, loss_meter=loss_meter
            ))

        runtimes[run] = runtime
        mIoU = np.mean(list(IoU.values()))
        print('mIoU---Val result: mIoU {:.4f}.'.format(mIoU))
        for class_ in cls_union:
            print("Class {} : {:.4f}".format(class_, IoU[class_]))

        val_IoUs[run] = mIoU
        val_losses[run] = loss_meter.avg

    print('Average mIoU over {} runs --- {:.4f}.'.format(args.n_runs, val_IoUs.mean()))
    print('Average runtime / run --- {:.4f}.'.format(runtimes.mean()))

    return val_IoUs.mean(), val_losses.mean()


def evaluate_cla(
        args: argparse.Namespace,
        val_loader: torch.utils.data.DataLoader,
        model: nn.Module,
        transformer: nn.Module,
        trans_cla: nn.Module,
        fusion: nn.Module,
    ) -> Tuple[torch.tensor, torch.tensor]:

    # To accumulate the losses and metrics over the entire validation set
    val_Iou_meter = AverageMeter()

    model.eval()
    transformer.eval()
    trans_cla.eval()
    fusion.eval() 

    pbar = tqdm(val_loader, desc="Validation")
    for batch in pbar:
        # Unpack the batch
        qry_img, q_label, spprt_imgs, s_label, subcls, _, _ = batch

        spprt_imgs = spprt_imgs.to('cuda', non_blocking=True)
        s_label = s_label.to('cuda', non_blocking=True)
        q_label = q_label.to('cuda', non_blocking=True)
        qry_img = qry_img.to('cuda', non_blocking=True)
        if spprt_imgs.shape[1] == 1:
            spprt_imgs_reshape = spprt_imgs.squeeze(0).expand(
                2, 3, args.image_size, args.image_size)
        else:
            spprt_imgs_reshape = spprt_imgs.squeeze(0)  # [n_shots, 3, img_size, img_size]

        # ====== Phase 1: Process support set and query images ======
        f_s = model.extract_features(spprt_imgs_reshape)  # [b, c, h, w]
        f_q = model.extract_features(qry_img)  # [b, c, h, w]
        f_q = F.normalize(f_q, dim=1)

        # Weights of the classifier (pretrained on support set)
        binary_cls = nn.Conv2d(
            args.bottleneck_dim, args.num_classes_tr, kernel_size=1, bias=False
        ).cuda()
        optimizer = optim.SGD(binary_cls.parameters(), lr=args.cls_lr)

        # Dynamic class weights
        s_label_arr = s_label.cpu().numpy().copy()  # [n_task, n_shots, img_size, img_size]
        back_pix = np.where(s_label_arr == 0)
        target_pix = np.where(s_label_arr == 1)

        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, len(back_pix[0]) / len(target_pix[0])]).cuda(),
            ignore_index=255
        )

        with torch.no_grad():
            f_s = model.extract_features(spprt_imgs.squeeze(0))  # [n_task, n_shots, c, h, w]

        for index in range(args.adapt_iter):
            output_support = binary_cls(f_s)
            output_support = F.interpolate(
                output_support, size=s_label.size()[2:],
                mode='bilinear', align_corners=True
            )
            s_loss = criterion(output_support, s_label.squeeze(0))
            optimizer.zero_grad()
            s_loss.backward()
            optimizer.step()

        weights_cls = binary_cls.weight.data
        weights_cls_reshape = weights_cls.squeeze().unsqueeze(0).expand(
            args.batch_size, 2, weights_cls.shape[1]
        )  # [b, 2, c]

        # Update classifier weights with transformer
        updated_weights_cls = transformer(weights_cls_reshape, f_q, f_q)  # [b, 2, c]

        # Process positive and negative samples from the support set
        s_pos = mask_pooling_single_class(f_s, s_label, 1)
        s_neg = mask_pooling_single_class(f_s, s_label, 0)
        s_pos = F.normalize(s_pos, dim=1)
        s_neg = F.normalize(s_neg, dim=1)

        f_q_pos = trans_cla(f_q, s_pos)
        f_q_neg = trans_cla(f_q, s_neg)
        f_q_fused = fusion(f_q, f_q_pos, f_q_neg).view(args.batch_size, args.bottleneck_dim, -1)

        # Prediction based on updated classifier weights
        q_pred_fused = torch.matmul(updated_weights_cls, f_q_fused).view(
            args.batch_size, 2, f_q.shape[-2], f_q.shape[-1]
        )

        q_logits = F.interpolate(
            q_pred_fused, size=q_label.shape[1:],
            mode='bilinear', align_corners=True
        )

        intersection, union, target = intersectionAndUnionGPU(
            q_logits.argmax(1), q_label, args.num_classes_tr, 255
        )

        # Update mIoU meter
        val_Iou_meter.update(intersection.sum() / (union.sum() + 1e-12), q_label.size(0))

    # Return the final averaged loss and mIoU for the validation set
    return val_Iou_meter.avg


def validate_supervised(
        args: argparse.Namespace,
        val_loader: torch.utils.data.DataLoader,
        model: DDP,
        # transformer: DDP
    ) -> Tuple[torch.tensor, torch.tensor]:
    """
    validate fully-supervised PSPNet on trav dataset. 
    """
    print('==> Start testing')

    model.eval()
    # transformer.eval()
    # n_episodes = int(args.test_num / args.batch_size_val)

    # ====== Metrics initialization  ======
    # H, W = args.image_size, args.image_size
    c = model.bottleneck_dim
    if args.image_size == 473:
        h = args.image_size
        w = args.image_size
    else:
        h = model.feature_res[0]
        w = model.feature_res[1]

    runtimes = torch.zeros(args.n_runs)
    val_IoUs = np.zeros(args.n_runs)
    val_losses = np.zeros(args.n_runs)

    # ====== Perform the runs  ======
    for run in range(args.n_runs):  # 1

        # ====== Initialize the metric dictionaries ======
        loss_meter = AverageMeter()
        iter_num = 0
        cls_intersection = defaultdict(int)  # Default value is 0
        cls_union = defaultdict(int)
        IoU = defaultdict(int)
        runtime = 0

        # for e in range(nb_episodes):  # 10
        t0 = time.time()
        logits_q = torch.zeros(args.test_num, 1, args.num_classes_tr, h, w).to('cuda') # [100,1,2,60,60]
        gt_q = 255 * torch.ones(
            args.batch_size_val, 1, args.image_size,args.image_size
        ).long().to('cuda')  # [100,1,473,473]
        classes = []  # All classes considered in the tasks

        # ====== Process each task separately ======
        # Batch size val is 50 here.

        # for i in range(args.batch_size_val):
        pbar = tqdm(val_loader)
        for i, batch in enumerate(pbar):
            # try:
            qry_img, q_label, spprt_imgs, s_label, subcls, spprt_oris, qry_oris = batch
            # except:
            #     iter_loader = iter(val_loader)
            #     qry_img, q_label, spprt_imgs, s_label, subcls, spprt_oris, qry_oris = iter_loader.next()
            iter_num += 1

            spprt_imgs = spprt_imgs.to('cuda', non_blocking=True)  # [1,1,3,473,473]
            s_label = s_label.to('cuda', non_blocking=True)  # [1,1,473,473]

            q_label = q_label.to('cuda', non_blocking=True)  # [1,473,473]
            qry_img = qry_img.to('cuda', non_blocking=True)  # [1,3,473,473]

            # ====== Phase 1: Train a new binary classifier on support samples. ======
            # binary_classifier = nn.Conv2d(
            #     args.bottleneck_dim, args.num_classes_tr, kernel_size=1, bias=False
            # ).cuda()

            # optimizer = optim.SGD(binary_classifier.parameters(), lr=args.cls_lr)

            # # Dynamic class weights
            # s_label_arr = s_label.cpu().numpy().copy()  # [n_task, n_shots, img_size, img_size]
            # back_pix = np.where(s_label_arr == 0)
            # target_pix = np.where(s_label_arr == 1)

            # criterion = nn.CrossEntropyLoss(
            #     weight=torch.tensor([1.0, len(back_pix[0]) / len(target_pix[0])]).cuda(),
            #     ignore_index=255
            # )

            # with torch.no_grad():
            #     logits = model(qry_img)  # [n_task, n_shots, c, h, w]

            # for index in range(args.adapt_iter):
            #     output_support = binary_classifier(f_s)
            #     output_support = F.interpolate(
            #         output_support, size=s_label.size()[2:],
            #         mode='bilinear', align_corners=True
            #     )
            #     s_loss = criterion(output_support, s_label.squeeze(0))
            #     optimizer.zero_grad()
            #     s_loss.backward()
            #     optimizer.step()

            # ====== Phase 2: Update classifier's weights with old weights and query features. ======
            with torch.no_grad():
                pred_q = model(qry_img)  # [n_task, c, h, w]
            #     f_q = F.normalize(f_q, dim=1)

            #     weights_cls = binary_classifier.weight.data  # [2, c, 1, 1]

            #     weights_cls_reshape = weights_cls.squeeze().unsqueeze(0).expand(
            #         f_q.shape[0], 2, 512
            #     )  # [1, 2, c]

            #     updated_weights_cls = transformer(weights_cls_reshape, f_q, f_q)  # [1, 2, c]

            #     # Build a temporary new classifier for prediction
            #     Pseudo_cls = nn.Conv2d(
            #         args.bottleneck_dim, args.num_classes_tr, kernel_size=1, bias=False
            #     ).cuda()

            #     # Initialize the weights with updated ones
            #     Pseudo_cls.weight.data = torch.as_tensor(
            #         updated_weights_cls.squeeze(0).unsqueeze(2).unsqueeze(3)
            #     )

            #     pred_q = Pseudo_cls(f_q)

            logits_q[i] = pred_q.detach()
            gt_q[i, 0] = q_label
            classes.append([class_.item() for class_ in subcls])
            pbar.set_description(f'iter: {i}')

        t1 = time.time()
        runtime += t1 - t0

        logits = logits_q.detach()

        intersection, union, _ = batch_intersectionAndUnionGPU(
            logits, gt_q, 2
        )

        intersection, union = intersection.cpu(), union.cpu()

        # ====== Log metrics ======
        criterion_standard = nn.CrossEntropyLoss(ignore_index=255)
        loss = criterion_standard(logits.squeeze(1), gt_q.squeeze(1))
        loss_meter.update(loss.item())
        for i, task_classes in enumerate(classes):
            for j, class_ in enumerate(task_classes):
                cls_intersection[class_] += intersection[i, 0, j + 1]  # Do not count background
                cls_union[class_] += union[i, 0, j + 1]

        for class_ in cls_union:
            IoU[class_] = cls_intersection[class_] / (cls_union[class_] + 1e-10)

        if (iter_num % 200 == 0):
            mIoU = np.mean([IoU[i] for i in IoU])
            print('Test: [{}/{}] '
                    'mIoU {:.4f} '
                    'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '.format(
                iter_num, args.test_num, mIoU, loss_meter=loss_meter
            ))

        runtimes[run] = runtime
        mIoU = np.mean(list(IoU.values()))
        print('mIoU---Val result: mIoU {:.4f}.'.format(mIoU))
        for class_ in cls_union:
            print("Class {} : {:.4f}".format(class_, IoU[class_]))

        val_IoUs[run] = mIoU
        val_losses[run] = loss_meter.avg

    print('Average mIoU over {} runs --- {:.4f}.'.format(args.n_runs, val_IoUs.mean()))
    print('Average runtime / run --- {:.4f}.'.format(runtimes.mean()))

    return val_IoUs.mean(), val_losses.mean()


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpus)

    if args.debug:
        args.test_num = 500
        args.n_runs = 2

    world_size = len(args.gpus)
    distributed = world_size > 1
    args.distributed = distributed
    args.port = find_free_port()
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)