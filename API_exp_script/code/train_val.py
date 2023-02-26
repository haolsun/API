"""Training Script"""
import os
import shutil
import torch
import torch.nn as nn
import torch.optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
# from torchviz import make_dot

from models.fewshot import FewShotSeg
from util.utils import set_seed
from config import ex

from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from common.logger import Logger, AverageMeter

#os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def train(config, epoch, model, dataloader, optimizer, training, n_pro, n_mk):
    r""" Train  """

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)

    ged_value_sum = 0.
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

        elif 'query_mask_a' in batch:
            pred_mask_a = query_pred.argmax(dim=2)
            batch_ = batch.copy()
            # cross energy
            iou_cross = []
            for i in range(batch['query_mask_a'].shape[1]):
                for j in range(n_pro * n_mk):
                    batch_['query_mask'] = batch['query_mask_a'][:,i,:,:]
                    area_inter, area_union = Evaluator.classify_prediction(pred_mask_a[:,j,:,:], batch_)
                    iou = (area_inter.float() / \
                          torch.max(torch.stack([area_union, torch.ones_like(area_union)]), dim=0)[0])[1]
                    iou_cross.append(1.0-iou)
            cross_energy = torch.stack(iou_cross).mean(dim=0)

            # inner energy mask
            iou_inner_mask = []
            for i in range(batch['query_mask_a'].shape[1]):
                for j in range(batch['query_mask_a'].shape[1]):
                    batch_['query_mask'] = batch['query_mask_a'][:, i, :, :]
                    area_inter, area_union = Evaluator.classify_prediction(batch['query_mask_a'][:, j, :, :], batch_)
                    iou = (area_inter.float() / \
                           torch.max(torch.stack([area_union, torch.ones_like(area_union)]), dim=0)[0])[1]
                    iou_inner_mask.append(1.0-iou)
            inner_mask_energy = torch.stack(iou_inner_mask).mean(dim=0)

            # inner energy pred
            iou_inner_pred = []
            for i in range(n_pro * n_mk):
                for j in range(n_pro * n_mk):
                    batch_['query_mask'] = pred_mask_a[:,i,:,:]
                    area_inter, area_union = Evaluator.classify_prediction(pred_mask_a[:,j,:,:], batch_)
                    iou = (area_inter.float() / \
                           torch.max(torch.stack([area_union, torch.ones_like(area_union)]), dim=0)[0])[1]
                    iou_inner_pred.append(1.0-iou)
            inner_pred_energy = torch.stack(iou_inner_pred).mean(dim=0)

            ged_value = 2*cross_energy - inner_pred_energy - inner_mask_energy
            ged_value_sum += ged_value.mean()


        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.sum().detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()



    return avg_loss, miou, fb_iou, ged_value_sum/(idx+1)

@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')


    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = False
    #torch.cuda.set_device(device=_config['gpu_id'])
    #torch.set_num_threads(1)

    Evaluator.initialize()

    _log.info('###### Create model ######')
    logging = open(f'{_run.observers[0].dir}/log.txt', 'w')
    # device = torch.device('cuda:0')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FewShotSeg(encoder=_config['model']['encoder'], out_dim=_config['n_ways']+1)
    # model = nn.DataParallel(device_ids=[0,1])
    # model.cuda()
    model = nn.DataParallel(model) #multi-gpu
    model.to(device)



    _log.info('###### Set optimizer ######')
    params = [{'params': model.module.decoder.parameters(), 'lr': _config['lr']},
              {'params': model.module.inference_ft.parameters(), 'lr': _config['lr']},
              {'params': model.module.inference.parameters(), 'lr': _config['lr']},
              {'params': model.module.prior.parameters(), 'lr': _config['lr']},
              {'params': model.module.prior_ft.parameters(), 'lr': _config['lr']},
              {'params': model.module.encoder.parameters(), 'lr': .01 * _config['lr']}]
    optimizer = torch.optim.Adam(params, lr=_config['lr'], weight_decay=_config['weight_decay'])
    # scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=0.3)


    _log.info('###### Load data ######')
    # data_name = _config['dataset']
    # Dataset initialization
    FSSDataset.initialize(img_size=400, datapath=_config['dataset_path'], use_original_imgsize=False)
    dataloader_trn = FSSDataset.build_dataloader(_config['dataset'], _config['batch_size'], 8, _config['label_sets'], 'trn', _config['n_shots'])
    dataloader_val = FSSDataset.build_dataloader(_config['dataset'], _config['batch_size'], 8, _config['label_sets'], 'val', _config['n_shots'])

    # Train
    n_sample_tr_pro, n_sample_tr_mk, n_sample_test_pro, n_sample_test_mk = 1, 1, 2, 2
    best_val_miou = float('-inf')
    # best_val_loss = float('inf')
    best_epoch = 0
    for epoch in range(_config['n_iters']):
        trn_loss, trn_miou, trn_fb_iou, trn_ged = train(_config, epoch, model, dataloader_trn, optimizer, True, n_sample_tr_pro, n_sample_tr_mk)
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou, val_ged = train(_config, epoch, model, dataloader_val, optimizer, False, n_sample_test_pro, n_sample_test_mk)
            Logger.info(f'val_ged: {val_ged}')

        # Save the best model
        if val_miou > best_val_miou:
            best_val_miou, best_epoch = val_miou, epoch
            _log.info('###### Taking snapshot for best ######')
            torch.save(model.state_dict(),
                       os.path.join(f'{_run.observers[0].dir}/snapshots', 'best_val.pth'))

        logging.write(f'epoch: {epoch}  trn_loss: {trn_loss}   trn_miou: {trn_miou}   trn_fb_iou: {trn_fb_iou}\n')
        logging.write(f'\t\t  val_loss: {val_loss}  --- val_miou: {val_miou}  val_fb_iou: {val_fb_iou} val_ged: {val_ged}\n')

    Logger.info('==================== Finished Training ====================')
    Logger.info(f'Best val_miou: {best_val_miou} at epoch {best_epoch}')
    logging.write(f'Best val_miou: {best_val_miou} at epoch {best_epoch}')
    logging.close()

