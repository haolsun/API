"""Training Script"""
import os
import shutil
import torch
import torch.nn as nn
import torch.optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn

from models.fewshot import FewShotSeg
from util.utils import set_seed
from config import ex

from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from common.logger import Logger, AverageMeter

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def train(config, epoch, model, dataloader, optimizer, training, n_pro, n_mk):
    r""" Train  """

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        batch = utils.to_cuda(batch)
        # Prepare input
        support_images = [[batch['support_imgs'][:,i,:,:,:] for i in range(config['n_shots'])]]
        support_fg_mask = [[batch['support_masks'][:,i,:,:] for i in range(config['n_shots'])]]
        support_bg_mask = [[1.0 - 1.0 * batch['support_masks'][:,i,:,:] for i in range(config['n_shots'])]]
        query_images = [batch['query_img']]
        query_labels = batch['query_mask']

        # Forward and Backward

        # 1. API Networks forward pass
        query_pred, kl_loss, _ = model(support_images, support_fg_mask, support_bg_mask,
                                       query_images, query_labels, train=training,
                                       n_sample_pro=n_pro, n_sample_mk=n_mk)
        pred_mask = query_pred.mean(axis=1).argmax(dim=1) # error for multi-GPU

        # 2. Compute loss & update model parameters
        loss = model.module.loss(config, 1.0 * epoch/config['n_iters'], query_labels.long(), query_pred, kl_loss, n_pro, n_mk)
        if training:
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.sum().detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou

@ex.automain
def main(_run, _config, _log):
    # if _run.observers:
    #     os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
    #     for source_file, _ in _run.experiment_info['sources']:
    #         os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
    #                     exist_ok=True)
    #         _run.observers[0].save_file(source_file, f'source/{source_file}')
    #     shutil.rmtree(f'{_run.observers[0].basedir}/_sources')


    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = False
    Evaluator.initialize()

    _log.info('###### Create model ######')
    logging = open(f'{_run.observers[0].dir}/log_test.txt', 'w')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FewShotSeg(encoder=_config['model']['encoder'], out_dim=_config['n_ways']+1)
    model = nn.DataParallel(model) #multi-gpu
    model.to(device)


    _log.info('###### Load data ######')
    # Dataset initialization
    FSSDataset.initialize(img_size=400, datapath=_config['dataset_path'], use_original_imgsize=False)
    dataloader_val = FSSDataset.build_dataloader(_config['dataset'], _config['batch_size'], 8, _config['label_sets'],
                                                 'test', _config['n_shots'])

    # load snapshot
    if _config['load_snapshot'] is not None:
        model.load_state_dict(torch.load(_config['load_snapshot']))

    # Test
    n_sample_tr_pro, n_sample_tr_mk, n_sample_test_pro, n_sample_test_mk = 1, 1, 5, 5
    with torch.no_grad():
        val_loss, val_miou, val_fb_iou = train(_config, 0, model, dataloader_val, None, False,
                                               n_sample_test_pro, n_sample_test_mk)
        logging.write(f'test_loss: {val_loss}  test_miou: {val_miou}  test_fb_iou: {val_fb_iou}\n')
    logging.close()

