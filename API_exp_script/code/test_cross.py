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

def test(config, model, dataloader, training, n_pro, n_mk):
    r""" Test  """

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

        # 2. Compute loss & ged
        loss = model.module.loss(config, 1.0 * 0 / config['n_iters'], query_labels.long(), query_pred, kl_loss, n_pro,
                                 n_mk)

        pred_mask_a = query_pred.argmax(dim=2)
        batch_ = batch.copy()
        # cross energy
        iou_cross = []
        for i in range(batch['query_mask_a'].shape[1]):
            for j in range(n_pro * n_mk):
                batch_['query_mask'] = batch['query_mask_a'][:, i, :, :]
                area_inter, area_union = Evaluator.classify_prediction(pred_mask_a[:, j, :, :], batch_)
                iou = (area_inter.float() / \
                       torch.max(torch.stack([area_union, torch.ones_like(area_union)]), dim=0)[0])[1]
                iou_cross.append(1.0 - iou)
        cross_energy = torch.stack(iou_cross).mean(dim=0)




        ged_value_sum += cross_energy.mean()


        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.sum().detach().clone())
        average_meter.write_process(idx, len(dataloader), 0, write_batch_idx=50)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', 0)
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
    model.eval()
    model = nn.DataParallel(model) #multi-gpu
    model.to(device)

    model.load_state_dict(torch.load('./runs/API_LIDC_align_encoder_sets_0_1way_1shot_train/2/snapshots/best_val.pth'))

    _log.info('###### Load data ######')
    # data_name = _config['dataset']
    # Dataset initialization
    FSSDataset.initialize(img_size=400, datapath=_config['dataset_path'], use_original_imgsize=False)
    dataloader_tst = FSSDataset.build_dataloader(_config['dataset'], _config['batch_size'], 8, _config['label_sets'], 'test', _config['n_shots'])

    # test
    n_sample_test_pro, n_sample_test_mk = 3, 3
    with torch.no_grad():
        val_loss, val_miou, val_fb_iou, val_ged = test(_config, model, dataloader_tst, False,
                                                    n_sample_test_pro, n_sample_test_mk)

    Logger.info('==================== Finished Testing ====================')
    Logger.info(f'val_miou: {val_miou}  val_fb_iou: {val_fb_iou}  val_ged: {val_ged}')
    logging.write(f'val_miou: {val_miou}  val_fb_iou: {val_fb_iou}  val_ged: {val_ged}')
    logging.close()

