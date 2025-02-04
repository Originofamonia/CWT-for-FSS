"""
FSS+CLA
"""

import os
# import sys
# import random
import numpy as np
import torch
# import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.optim as optim
# from collections import defaultdict
# from typing import Dict
# from torch import Tensor
# from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
# import torch.multiprocessing as mp
import argparse
from typing import Tuple
from tqdm import tqdm

from model.pspnet import get_model_fs
from model.transformer import MultiHeadAttentionOne, AttentionExtractor, ConvFusion
from optimizer import get_optimizer, get_scheduler
from dataset.dataset import trav_train_loader, trav_val_loader
from util import intersectionAndUnionGPU, get_model_dir, AverageMeter, get_model_dir_trans
# from util import setup, cleanup, to_one_hot, batch_intersectionAndUnionGPU, find_free_port
from test import evaluate_cla
from util import load_cfg_from_cfg_file, merge_cfg_from_list, mask_pooling_single_class

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # keep this


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Training classifier weight transformer')
    parser.add_argument('--config', type=str, default=f'config_files/trav.yaml', help='config file')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg


def main_worker(args: argparse.Namespace) -> None:
    print(f"==> Running process rank {args.device}.")
    print(args)

    # ====== Model + Optimizer ======
    model = get_model_fs(args).to('cuda')

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

        # Fix the backbone layers
        for param in model.layer0.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = False
        for param in model.ppm.parameters():
            param.requires_grad = False
        for param in model.bottleneck.parameters():
            param.requires_grad = False

    # ====== Transformer ======
    trans_dim = args.bottleneck_dim

    transformer = MultiHeadAttentionOne(
        args.heads, trans_dim, trans_dim, trans_dim, dropout=0.5
    ).to('cuda')

    trans_cla = AttentionExtractor(
        args.heads, trans_dim, trans_dim, trans_dim
    ).to('cuda')

    fusion = ConvFusion(trans_dim).to('cuda')

    optimizer_transformer = get_optimizer(
        args,
        [dict(params=transformer.parameters(), lr=args.trans_lr * args.scale_lr),
         dict(params=trans_cla.parameters(), lr=args.trans_lr * args.scale_lr),
         dict(params=fusion.parameters(), lr=args.trans_lr * args.scale_lr),]
    )

    trans_save_dir = get_model_dir_trans(args)

    # ====== Data  ======
    train_loader, train_sampler = trav_train_loader(args)
    episodic_val_loader, _ = trav_val_loader(args)

    # ====== Metrics initialization ======
    max_val_mIoU = 0.
    log_iter = args.iter_per_epoch

    # ====== Training  ======
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        _, _ = do_epoch(
            args=args,
            train_loader=train_loader,
            model=model,
            transformer=transformer,
            trans_cla=trans_cla,
            fusion=fusion,
            optimizer_trans=optimizer_transformer,
            epoch=epoch,
            log_iter=log_iter,
        )
        if epoch % log_iter == 0:
            val_Iou = evaluate_cla(
                args=args,
                val_loader=episodic_val_loader,
                model=model,
                transformer=transformer,
                trans_cla=trans_cla,
                fusion=fusion,
            )

            if val_Iou.item() > max_val_mIoU:
                max_val_mIoU = val_Iou.item()

                os.makedirs(trans_save_dir, exist_ok=True)
                filename_transformer = os.path.join(trans_save_dir, f'best.pth')

                if args.save_models:
                    print('Saving checkpoint to: ' + filename_transformer)

                    torch.save(
                        {
                            'epoch': epoch,
                            'state_dict': transformer.state_dict(),
                            'optimizer': optimizer_transformer.state_dict()
                        },
                        filename_transformer
                    )

            print(f"curr mIoU: {val_Iou.item():.3f}; Max_mIoU = {max_val_mIoU:.3f}")
        
    val_Iou = evaluate_cla(
        args=args,
        val_loader=episodic_val_loader,
        model=model,
        transformer=transformer,
        trans_cla=trans_cla,
        fusion=fusion,
    )
    print(f"Final mIoU: {val_Iou.item():.3f}; Max_mIoU = {max_val_mIoU:.3f}")

    if args.save_models:
        filename_transformer = os.path.join(trans_save_dir, 'final.pth')
        torch.save(
            {
                'epoch': args.epochs,
                'state_dict': transformer.state_dict(),
                'optimizer': optimizer_transformer.state_dict()
             },
            filename_transformer
        )


def main_process(args: argparse.Namespace) -> bool:
    if args.distributed:
        rank = 'cuda'
        if rank == 0:
            return True
        else:
            return False
    else:
        return False


def do_epoch(
        args: argparse.Namespace,
        train_loader: torch.utils.data.DataLoader,
        model: nn.Module,
        transformer: nn.Module,
        trans_cla: nn.Module,
        fusion: nn.Module,
        optimizer_trans: torch.optim.Optimizer,
        epoch: int,
        log_iter: int
    ) -> Tuple[torch.tensor, torch.tensor]:

    loss_meter = AverageMeter()
    train_losses = torch.zeros(log_iter).to('cuda')
    train_Ious = torch.zeros(log_iter).to('cuda')

    model.train()
    transformer.train()
    trans_cla.train()
    pbar = tqdm(train_loader)
    for i, batch in enumerate(pbar):
        qry_img, q_label, spprt_imgs, s_label, _, _, _ = batch

        spprt_imgs = spprt_imgs.to('cuda', non_blocking=True)
        s_label = s_label.to('cuda', non_blocking=True)
        q_label = q_label.to('cuda', non_blocking=True)
        qry_img = qry_img.to('cuda', non_blocking=True)

        # ====== Phase 1: Train the binary classifier on support samples ======

        # Keep the batch size as 1.
        if spprt_imgs.shape[1] == 1:
            spprt_imgs_reshape = spprt_imgs.squeeze(0).expand(
                2, 3, args.image_size, args.image_size
            )
            s_label_reshape = s_label.squeeze(0).expand(
                2, args.image_size, args.image_size
            ).long()
        else:
            spprt_imgs_reshape = spprt_imgs.squeeze(0)  # [n_shots, 3, img_size, img_size]
            s_label_reshape = s_label.squeeze(0).long() # [n_shots, img_size, img_size]

        binary_cls = nn.Conv2d(
            args.bottleneck_dim, args.num_classes_tr, kernel_size=1, bias=False
        ).cuda()

        optimizer = optim.SGD(binary_cls.parameters(), lr=args.cls_lr)

        # Dynamic class weights
        s_label_arr = s_label.cpu().numpy().copy()  # [b, n_shots, img_size, img_size]
        back_pix = np.where(s_label_arr == 0)
        target_pix = np.where(s_label_arr == 1)

        if len(back_pix[0]) == 0 or len(target_pix[0]) == 0:
            continue  # skip bad support set
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, len(back_pix[0]) / len(target_pix[0])]).cuda(),
            ignore_index=255
        )

        with torch.no_grad():
            f_s = model.extract_features(spprt_imgs_reshape)  # [b, c, h, w]

        for index in range(args.adapt_iter):
            output_support = binary_cls(f_s)
            output_support = F.interpolate(
                output_support, size=s_label.size()[2:],
                mode='bilinear', align_corners=True
            )
            s_loss = criterion(output_support, s_label_reshape)
            optimizer.zero_grad()
            s_loss.backward()
            optimizer.step()

        # ====== Phase 2: Train the transformer to update the classifier's weights ======
        # Inputs of the transformer: weights of classifier trained on support sets, features of the query sample.

        # Dynamic class weights used for query image only during training
        q_label_arr = q_label.cpu().numpy().copy()  # [b, img_size, img_size]
        q_back_pix = np.where(q_label_arr == 0)
        q_target_pix = np.where(q_label_arr == 1)

        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, len(q_back_pix[0]) / max(len(q_target_pix[0]), 10)]).cuda(),
            ignore_index=255
        )

        model.eval()
        with torch.no_grad():
            f_q = model.extract_features(qry_img)  # [b, c, h, w]
            f_q = F.normalize(f_q, dim=1)

        # Weights of the classifier.
        weights_cls = binary_cls.weight.data

        weights_cls_reshape = weights_cls.squeeze().unsqueeze(0).expand(
            args.batch_size, 2, weights_cls.shape[1]
        )  # [b, 2, c]

        # Update the classifier's weights with transformer
        updated_weights_cls = transformer(weights_cls_reshape, f_q, f_q)  # [b, 2, c]

        s_pos = mask_pooling_single_class(f_s, s_label, 1)
        s_neg = mask_pooling_single_class(f_s, s_label, 0)
        s_pos = F.normalize(s_pos, dim=1)
        s_neg = F.normalize(s_neg, dim=1)

        f_q_pos = trans_cla(f_q, s_pos)
        f_q_neg = trans_cla(f_q, s_neg)
        f_q_fused = fusion(f_q, f_q_pos, f_q_neg).view(args.batch_size, args.bottleneck_dim, -1)
        q_pred_fused = torch.matmul(updated_weights_cls, f_q_fused).view(
            args.batch_size, 2, f_q.shape[-2], f_q.shape[-1]
        )
        q_logits = F.interpolate(
            q_pred_fused, size=q_label.shape[1:],
            mode='bilinear', align_corners=True
        )
        
        loss_total = criterion(q_logits, q_label.long())

        optimizer_trans.zero_grad()
        loss_total.backward()
        optimizer_trans.step()

        # Print loss and mIoU
        intersection, union, target = intersectionAndUnionGPU(
            q_logits.argmax(1), q_label, args.num_classes_tr, 255
        )

        if args.distributed:
            dist.all_reduce(loss_total)
            dist.all_reduce(intersection)
            dist.all_reduce(union)
            dist.all_reduce(target)

        mIoU = (intersection / (union + 1e-10)).mean()
        loss_meter.update(loss_total.item())

        train_losses[i] = loss_meter.avg
        train_Ious[i] = mIoU
        
        pbar.set_description(f'e: {epoch}, iter: {i}, miou: {mIoU:.3f}, loss: {loss_meter.avg:.3f}')

    print(F'Epoch {epoch}: mIoU {train_Ious.mean():.3f}, loss {train_losses.mean():.3f}')

    return train_Ious, train_losses


if __name__ == "__main__":
    args = parse_args()
    main_worker(args)