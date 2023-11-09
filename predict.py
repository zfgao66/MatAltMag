"""
load weights of classifier model and predict the probability of
candidate materials.

"""
import config as cfg
from kogger import Logger
import pprint
from dataset import PredictCIFData
from dataset_helper import collate_pool
from torch.utils.data import DataLoader
from model import CrystalGraph
from accelerate import Accelerator
from utils import AverageMeter
import time
import pandas as pd
import torch

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def predict(accelerator, data_loader, model, logger, config):
    # switch to eval mode
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    cif_ids = []
    cif_probs = []

    end = time.time()
    for batch_idx, (inputs, target, cif_id) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # input: list, len=4
        # target: [b, ]
        # cif_id: [b, ]
        atom_fea, nbr_fea, nbr_fea_idx, degree, crystal_atom_idx = inputs
        with torch.no_grad():
            output = model.predict(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)  # [b, features]
            label1_probs = output.detach().cpu()[:, 1]  # [b, ]

        cif_ids = cif_ids + cif_id
        cif_probs = cif_probs + label1_probs.tolist()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if accelerator.is_main_process and batch_idx % config['log_batch_freq'] == 0:
            logger.info('Batch [{}/{}]\t BT {:.3f} ({:.3f})\t DT {:.3f} ({:.3f})\t'.format(batch_idx, len(data_loader), batch_time.val, batch_time.avg, data_time.val, data_time.avg))

    return cif_ids, cif_probs


def main():
    # load and set config
    args = cfg.get_parser().parse_args()
    config = cfg.load_config(yaml_filename=args.filename)
    config = cfg.process_config(config)

    accelerator = Accelerator()

    logger = Logger('PID %d' % accelerator.process_index)
    if accelerator.is_main_process:
        logger.info('Load config successfully!')
        logger.info(pprint.pformat(config))

    dataset = PredictCIFData(
        root_dir=config['root_dir'],
        processed_dir=config['processed_dir'],
        radius=config['radius'],
        max_num_nbr=config['max_num_nbr'],
        dmin=config['dmin'],
        step=config['step'],
        logger=logger
    )

    data_loader = DataLoader(
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

    model, data_loader = accelerator.prepare(
        model, data_loader
    )
    accelerator.load_state(input_dir=config['ckpt_path'])

    cif_ids, cif_probs = predict(
        accelerator=accelerator,
        model=model,
        data_loader=data_loader,
        config=config,
        logger=logger
    )

    df = pd.DataFrame({'id': cif_ids, 'prob': cif_probs})
    df.sort_values(by='prob', ascending=False, inplace=True)
    df.to_csv(config['output_file'], header=None, index=None)

    logger.info('Done!')

if __name__ == '__main__':
    main()
