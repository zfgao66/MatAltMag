"""
load the weights of encoder from auto-encoder model and train classifier model
on the fine-tuning datasets.

"""
import config as cfg
from kogger import Logger
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
import pprint
from dataset import TrainCIFData
from dataset_helper import collate_pool
from utils import AverageMeter
from model import CrystalGraph
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import time
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def train(accelerator, model, writer, train_loader, optimizer, scheduler, loss_func, config, logger):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    for epoch in range(1, config['epochs']+1):
        end = time.time()
        for batch_idx, (inputs, target, _) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # input: list, len=4
            # label: [b, ]
            atom_fea, nbr_fea, nbr_fea_idx, degree, crystal_atom_idx = inputs
            output = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)  # [b, features]
            loss = loss_func(output, target)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            with torch.no_grad():
                losses.update(loss.item(), target.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if accelerator.is_main_process and (epoch % config['log_epoch_freq'] == 0 or epoch == 1) and batch_idx % config['log_batch_freq'] == 0:
                logger.info('[Train] Epoch [{}/{}] [{}/{}]\t BT {:.3f} ({:.3f})\t DT {:.3f} ({:.3f})\t Loss {:.4e} ({:.4e})\t'.format(epoch, config['epochs'], batch_idx, len(train_loader), batch_time.val, batch_time.avg, data_time.val, data_time.avg, losses.val, losses.avg))

        scheduler.step()
        if accelerator.is_main_process:
            writer.add_scalar('Loss/train', losses.val, epoch)
        if accelerator.is_main_process and epoch % config['save_epoch_freq'] == 0:
            accelerator.save_state(output_dir=config['ckpt_path'])


def main():
    # load and set config
    args = cfg.get_parser().parse_args()
    config = cfg.load_config(yaml_filename=args.filename)
    config = cfg.process_config(config)

    # accelerator = Accelerator()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    writer = SummaryWriter(comment=config['comment'])

    logger = Logger('PID %d' % accelerator.process_index, file=config['log_file'])
    # logger = Logger('PID %d' % accelerator.process_index)
    if accelerator.is_main_process:
        logger.info('Load config successfully!')
        logger.info(pprint.pformat(config))

    # load data
    if accelerator.is_main_process:
        logger.info('Load data...')
    dataset = TrainCIFData(
        root_dir=config['root_dir'],
        processed_dir=config['processed_dir'],
        radius=config['radius'],
        ratio=config['ratio'],
        max_num_nbr=config['max_num_nbr'],
        dmin=config['dmin'],
        step=config['step'],
        logger=logger
    )
    # export_data(dataset)
    # exit(-1)

    train_loader = DataLoader(
        dataset=dataset,
        collate_fn=collate_pool,
        batch_size=config['batch_size'],
        shuffle=config['shuffle'],
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # build model
    inputs, _, _ = dataset[0]
    orig_atom_fea_len = inputs[0].shape[-1]
    nbr_fea_len = inputs[1].shape[-1]
    crystal_gnn_config = config['crystal_gnn_config']
    crystal_gnn_config['orig_atom_fea_len'] = orig_atom_fea_len
    crystal_gnn_config['nbr_fea_len'] = nbr_fea_len
    model = CrystalGraph(
        crystal_gnn_config=crystal_gnn_config,
        head_output_dim=config['head_output_dim'],
        drop_rate=config['drop_rate'],
        decoder_sample_size=config['sample_size'],
        device=accelerator.device
    )

    # train
    optimizer = optim.Adam(model.parameters(), config['lr'], weight_decay=config['weight_decay'])
    # optimizer = optim.SGD(model.parameters(), config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])
    scheduler = MultiStepLR(optimizer, milestones=config['lr_milestones'], gamma=0.1)
    model, train_loader, optimizer, scheduler = accelerator.prepare(
        model, train_loader, optimizer, scheduler
    )

    loss_func = nn.CrossEntropyLoss()

    # train
    if accelerator.is_main_process:
        logger.info('Train...')

    if config['load_pretrain']:
        if accelerator.is_main_process:
            logger.info('Load pre-train model...')
        accelerator.load_state(input_dir=config['pre_ckpt_path'])
    elif config['continuous_train']:
        if accelerator.is_main_process:
            logger.info('Continue train...')
        accelerator.load_state(input_dir=config['ckpt_path'])
    train(
        accelerator=accelerator,
        model=model,
        writer=writer,
        train_loader=train_loader,
        loss_func=loss_func,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        logger=logger
    )

    if accelerator.is_main_process:
        logger.info('Done!')


# def export_data(dataset):
#     posi_num = 22
#     nega_num = 22
#     save_dir = 'out/tmp'
#
#     for i in range(posi_num):
#         atom_fea, nbr_fea = dataset.dataset_posi[0][i].clone(), dataset.dataset_posi[1][i].clone()
#         torch.save(atom_fea, osp.join(save_dir, 'label1_atom_fea_{}.pkl'.format(i)))
#         torch.save(nbr_fea, osp.join(save_dir, 'label1_nbr_fea_{}.pkl'.format(i)))
#
#     for i in range(nega_num):
#         atom_fea, nbr_fea = dataset.dataset_nega[0][i].clone(), dataset.dataset_nega[1][i].clone()
#         torch.save(atom_fea, osp.join(save_dir, 'label0_atom_fea_{}.pkl'.format(i)))
#         torch.save(nbr_fea, osp.join(save_dir, 'label0_nbr_fea_{}.pkl'.format(i)))

if __name__ == '__main__':
    main()
