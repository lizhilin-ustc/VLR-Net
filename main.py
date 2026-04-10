import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import json
from collections import OrderedDict
import torch.nn.functional as F

from run_eval import evaluate
from utils import misc_utils
from dataset.dataset import ThumosFeature
from models.model import VLR_Net
from utils.loss import CrossEntropyLoss, GeneralizedCE
from config.config_thumos import Config, parse_args, class_dict

import time

np.set_printoptions(formatter={'float_kind': "{:.2f}".format})
torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)


def load_weight(net, config):
    if config.load_weight:
        model_file = os.path.join(config.model_path, "CAS_Only.pkl")
        print("loading from file for training: ", model_file)
        pretrained_params = torch.load(model_file)

        selected_params = OrderedDict()
        for k, v in pretrained_params.items():
            if 'base_module' in k:
                selected_params[k] = v

        model_dict = net.state_dict()
        model_dict.update(selected_params)
        net.load_state_dict(model_dict)

def my_load_weight(net, model_path):
    model_file = os.path.join(model_path, "CAS_Only.pkl")
    pretrained_params = torch.load(model_file)
    selected_params = OrderedDict()
    for k, v in pretrained_params.items():
        if 'fc' in k:
            selected_params[k] = v

    model_dict = net.state_dict()
    model_dict.update(selected_params)
    net.load_state_dict(model_dict)
    # net.load_state_dict(pretrained_params, strict=False)


def get_dataloaders(config):
    train_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode='train',
                      modal=config.modal, feature_fps=config.feature_fps,
                      num_segments=config.num_segments, len_feature=config.len_feature,
                      seed=config.seed, sampling='random', supervision='strong'),
        batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers)

    test_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode='test',
                      modal=config.modal, feature_fps=config.feature_fps,
                      num_segments=config.num_segments, len_feature=config.len_feature,
                      seed=config.seed, sampling='uniform', supervision='strong'),
        batch_size=1,
        shuffle=False, num_workers=config.num_workers)

    return train_loader, test_loader


def set_seed(config):
    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.deterministic = True
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.benchmark = False

def load_weight_inference(model_file, net):
    if model_file is not None:
        print("loading from file: ", model_file)
        net.load_state_dict(torch.load(model_file), strict=False)




class ThumosTrainer():
    def __init__(self, config):
        # config
        self.config = config

        # network
        # self.net = ModelFactory.get_model(config.model_name, config)
        self.net = VLR_Net(config.len_feature, config.num_classes)
        self.net = self.net.cuda()

        # data
        self.train_loader, self.test_loader = get_dataloaders(self.config)

        # loss, optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr, betas=(0.9, 0.999), weight_decay=0.0005)   # 0.0005
        self.criterion = CrossEntropyLoss()
        self.Lgce = GeneralizedCE(q=self.config.q_val)

        # parameters
        self.best_mAP = -1 # init
        self.step = 0
        self.best_acc = 0
        self.total_loss_per_epoch = 0
        # self.text_feat = torch.load(self.config.data_path + '/text_label/texts_THUMOS4_v2.pt').cuda()  # 4
        self.text_feat = torch.load(self.config.text_features).cuda() 


    def test(self):
        self.net.eval()
        with torch.no_grad():
            model_filename = "CAS_Only.pkl"
            self.config.model_file = os.path.join(self.config.model_path, model_filename)
            _mean_ap, test_acc = self.inference(self.net, self.config, self.test_loader, model_file=self.config.model_file)
            print("cls_acc={:.5f} map={:.5f}".format(test_acc*100, _mean_ap*100))


    def calculate_pesudo_target(self, batch_size, label, topk_indices):
        cls_agnostic_gt = []
        cls_agnostic_neg_gt = []
        for b in range(batch_size):
            label_indices_b = torch.nonzero(label[b, :])[:,0]
            topk_indices_b = topk_indices[b, :, label_indices_b] # topk, num_actions
            cls_agnostic_gt_b = torch.zeros((1, 1, self.config.num_segments)).cuda()

            # positive examples
            for gt_i in range(len(label_indices_b)):
                cls_agnostic_gt_b[0, 0, topk_indices_b[:, gt_i]] = 1
            cls_agnostic_gt.append(cls_agnostic_gt_b)

        return torch.cat(cls_agnostic_gt, dim=0)  # B, 1, num_segments

    def calculate_pesudo_target1(self, batch_size, topk_indices):
        cls_agnostic_gt = []
        for b in range(batch_size):
            topk_indices_b = topk_indices[b, :]          # [75, 1]
            cls_agnostic_gt_b = torch.zeros((1, 1, self.config.num_segments)).cuda()

            cls_agnostic_gt_b[0, 0, topk_indices_b[:, 0]] = 1
            cls_agnostic_gt.append(cls_agnostic_gt_b)

        return torch.cat(cls_agnostic_gt, dim=0)         # B, 1, num_segments
        
    def cas_pesduo_label(self, cas_score, label, threshold):
        pes_lab = cas_score * label.unsqueeze(1)
        pes_lab = torch.where(pes_lab>threshold, 1.0, 0.0)  # 0.75:47.27  0.8:47.33

        x = torch.ones(pes_lab.shape).cuda() / pes_lab.shape[-1]
        # x = x * (1-_label).unsqueeze(1)
        x1 = torch.sum(pes_lab,dim=-1)
        x1 = torch.where(x1>=1, 1.0, 0.0)
        x = x * (1-x1).unsqueeze(-1)

        pes_seg_lab = pes_lab + x
        pes_seg_lab = torch.where(pes_seg_lab >= 1, 1.0, pes_seg_lab)

        pes_seg_lab = pes_seg_lab / torch.sum(pes_seg_lab, dim=-1, keepdim=True)

        return pes_seg_lab
    
    def action_label(self, cas_score, label, threshold):
        pes_lab = cas_score * label.unsqueeze(1)
        pes_lab = torch.where(pes_lab>threshold, 1.0, 0.0)
        pes_lab = torch.sum(pes_lab, dim=-1)
        pes_lab = torch.where(pes_lab>=1, 1.0, 0.0)
        return pes_lab

    
    def evaluate(self, epoch=0):
        if self.step % self.config.detection_inf_step == 0:
            self.total_loss_per_epoch /= self.config.detection_inf_step

            with torch.no_grad():
                self.net = self.net.eval()
                mean_ap, test_acc = self.inference(self.net, self.config, self.test_loader, model_file=None)
                self.net = self.net.train()

            if mean_ap > self.best_mAP:
                self.best_mAP = mean_ap

                torch.save(self.net.state_dict(), os.path.join(self.config.model_path, "CAS_Only.pkl"))
                # torch.save(self.best_mAP, os.path.join(self.config.model_path, "best_mAP.txt"))
                with open(os.path.join(self.config.model_path, "best_mAP.txt"), 'a') as file:
                    write_str = '%f\n' % (self.best_mAP)
                    file.write(write_str)

            if test_acc > self.best_acc:
                self.best_acc = test_acc

            print("epoch={:5d} step={:5d} Loss={:.4f} cls_acc={:5.2f} best_acc={:5.2f} mean_ap={:5.2f} best_map={:5.2f}".format(
                    epoch, self.step, self.total_loss_per_epoch, test_acc * 100, self.best_acc * 100, mean_ap*100, self.best_mAP * 100))

            self.total_loss_per_epoch = 0


    def train(self):
        """Train the model"""
        # resume training
        load_weight(self.net, self.config)

        # training loop
        for epoch in range(self.config.num_epochs):
            for rf_feat, vl_feat, _label, temp_anno, vid_name, vid_num_seg in self.train_loader:
                batch_size = rf_feat.shape[0]
                rf_feat, vl_feat, _label = rf_feat.cuda(), vl_feat.cuda(), _label.cuda()
                self.optimizer.zero_grad()
                
                # forward pass
                cas, action_flow, action_rgb, action_vl, sim_matrix, sim_text_prj, sim_vl_prj, sim_prj = self.net(rf_feat, vl_feat, self.text_feat)
                
                # action fusion
                action_fusion = 0.4 * action_flow + 0.1 * action_rgb + 0.5 * action_vl
                combined_cas = (torch.softmax(cas, -1) + action_fusion.permute(0, 2, 1)) / 2
                _, topk_indices = torch.topk(combined_cas, self.config.num_segments // 8, dim=1)
                
                combined_cas1 = (torch.softmax(cas, -1) + (1 - action_fusion.permute(0, 2, 1))) / 2
                _, topk_indices1 = torch.topk(combined_cas1, self.config.num_segments // 8, dim=1)

                label = _label.clone().type(torch.int)
                topk_indices2 = topk_indices * label.unsqueeze(1) + topk_indices1 * (1 - label).unsqueeze(1)

                # get topk features
                cas_top = torch.mean(torch.gather(cas, 1, topk_indices2), dim=1)
                txt_top = torch.mean(torch.gather(sim_text_prj, 1, topk_indices2), dim=1)
                vl_top = torch.mean(torch.gather(sim_vl_prj, 1, topk_indices2), dim=1)
                
                # calculate pseudo targets
                _, topk_indices_agno = torch.topk(action_fusion.permute(0, 2, 1), 100, dim=1)
                topk_indices_agno = topk_indices_agno.repeat(1, 1, cas.shape[2])
                cls_agnostic_gt = self.calculate_pesudo_target1(cas.shape[0], topk_indices_agno).squeeze(1)

                # generate segment labels
                origin_seg_lab = self.cas_pesduo_label(sim_matrix, _label, self.config.alpha_h)
                bg_seg_lab = self.action_label(sim_matrix, _label, self.config.alpha_l)
                fore_seg_lab = self.action_label(sim_matrix, _label, self.config.alpha_h)
                cls_agnostic_gt = (cls_agnostic_gt * bg_seg_lab) + fore_seg_lab
                cls_agnostic_gt = torch.where(cls_agnostic_gt >= 1, 1.0, 0.0)

                # calculate losses
                mil_loss = self.criterion(cas_top, _label)
                act_loss = self.Lgce(action_flow.squeeze(1), cls_agnostic_gt) + \
                           self.Lgce(action_rgb.squeeze(1), cls_agnostic_gt) + \
                           self.Lgce(action_vl.squeeze(1), cls_agnostic_gt)
                mut_loss = F.mse_loss(action_rgb, action_flow) + \
                           F.mse_loss(action_flow, action_vl) + \
                           F.mse_loss(action_rgb, action_vl)
                loss_base = mil_loss + 1 * act_loss + 5 * mut_loss

                snip_loss = self.criterion(cas.reshape(-1, 20), origin_seg_lab.reshape(-1, 20))

                sim_text_prj = torch.softmax(sim_text_prj, dim=-1)
                sim_vl_prj = torch.softmax(sim_vl_prj, dim=-1)

                loss_vl = self.criterion(txt_top, _label) + self.criterion(vl_top, _label)
                kl_loss = F.kl_div(sim_text_prj.reshape(-1, 20).log(), sim_vl_prj.reshape(-1, 20).detach(), reduction='batchmean') + \
                          F.kl_div(sim_vl_prj.reshape(-1, 20).log(), sim_text_prj.reshape(-1, 20).detach(), reduction='batchmean')
                
                cost = loss_base + self.config.lambda1 * kl_loss + self.config.lambda2 * loss_vl + self.config.lambda3 * snip_loss

                # backpropagation
                cost.backward()
                self.optimizer.step()

                self.total_loss_per_epoch += cost.cpu().item()
                self.step += 1

                # evaluation
                self.evaluate(epoch=epoch)

    def inference(self, net, config, test_loader, model_file=None):
        """Run inference on the test set"""
        np.set_printoptions(formatter={'float_kind': "{:.4f}".format})

        with torch.no_grad():
            net.eval()

            # load weights
            load_weight_inference(model_file, net)

            final_res = {'version': 'VERSION 1.3', 'results': {},
                        'external_data': {'used': True, 'details': 'Features from I3D Network'}}

            num_correct = 0.
            num_total = 0.

            start_time = time.time()

            for rf_feat, vl_feat, _label, temp_anno, vid_name, vid_num_seg in test_loader:
                batch_size = rf_feat.shape[0]
                rf_feat = rf_feat.cuda()
                vl_feat = vl_feat.cuda()
                _label = _label.cuda()
                
                # forward pass
                cas, action_flow, action_rgb, action_vl, origin_score, sim_text_prj, sim_vl_prj, sim_prj = net(rf_feat, vl_feat, self.text_feat)
                
                # action fusion
                action_fusion = 0.4 * action_flow + 0.1 * action_rgb + 0.5 * action_vl
                combined_cas = (torch.softmax(cas, -1) + action_fusion.permute(0, 2, 1)) / 2
                _, topk_indices = torch.topk(combined_cas, config.num_segments // 8, dim=1)
                
                combined_cas1 = (torch.softmax(cas, -1) + (1 - action_fusion.permute(0, 2, 1))) / 2
                _, topk_indices1 = torch.topk(combined_cas1, self.config.num_segments // 8, dim=1)
                
                label = _label.clone().type(torch.int)
                topk_indices2 = topk_indices * label.unsqueeze(1) + topk_indices1 * (1 - label).unsqueeze(1)
                
                # get classification scores
                cas_top = torch.gather(cas, 1, topk_indices2)
                cas_top = torch.mean(cas_top, dim=1)
                score_supp = F.softmax(cas_top, dim=1)

                # calculate accuracy
                label_np = _label.cpu().numpy()
                score_np = score_supp[0, :].cpu().numpy()

                score_np[np.where(score_np < config.class_thresh)] = 0
                score_np[np.where(score_np >= config.class_thresh)] = 1

                if np.all(score_np == 0):
                    arg = np.argmax(score_supp[0, :].cpu().data.numpy())
                    score_np[arg] = 1

                correct_pred = np.sum(label_np == score_np, axis=1)
                num_correct += np.sum((correct_pred == config.num_classes).astype(np.float32))
                num_total += correct_pred.shape[0]

                # generate proposals
                pred = np.where(score_np > config.class_thresh)[0]
                if len(pred) != 0:
                    cas_pred = combined_cas[0].cpu().numpy()[:, pred]
                    cas_pred = np.reshape(cas_pred, (config.num_segments, -1, 1))
                    cas_pred = misc_utils.upgrade_resolution(cas_pred, config.scale)

                    proposal_dict = {}

                    for t in range(len(config.act_thresh)):
                        cas_temp = cas_pred.copy()
                        zero_location = np.where(cas_temp[:, :, 0] < config.act_thresh[t])
                        cas_temp[zero_location] = 0

                        seg_list = []
                        for c in range(len(pred)):
                            pos = np.where(cas_temp[:, c, 0] > 0)
                            seg_list.append(pos)

                        proposals = misc_utils.get_proposal_oic(seg_list, cas_pred.copy(), score_supp[0, :].cpu().data.numpy(),
                                                                pred, config.scale, vid_num_seg[0].cpu().item(), config.feature_fps,
                                                                config.num_segments, config.gamma)

                        for j in range(len(proposals)):
                            if not proposals[j]:
                                continue
                            class_id = proposals[j][0][0]

                            if class_id not in proposal_dict.keys():
                                proposal_dict[class_id] = []

                            proposal_dict[class_id] += proposals[j]

                    # apply NMS
                    final_proposals = []
                    for class_id in proposal_dict.keys():
                        final_proposals.append(misc_utils.basnet_nms(proposal_dict[class_id], config.nms_thresh,
                                                                    config.soft_nms, config.nms_alpha))

                    final_res['results'][vid_name[0]] = misc_utils.result2json(final_proposals)

            # calculate metrics
            test_acc = num_correct / num_total
            json_path = os.path.join(config.model_path, 'temp_result.json')

            end_time = time.time()
            print(f'Inference time (/video): {(end_time - start_time)/len(test_loader)}')

            # save results
            with open(json_path, 'w') as f:
                json.dump(final_res, f)

            # evaluate
            mean_ap, _ = evaluate(config.gt_path, json_path, None, tiou_thresholds=np.linspace(0.1, 0.7, 7), plot=False,
                                        subset='test', verbose=config.verbose)

            return mean_ap, test_acc

def main():
    args = parse_args()
    config = Config(args)
    set_seed(config)

    trainer = ThumosTrainer(config)

    if args.inference_only:
        trainer.test()
    else:
        trainer.train()


if __name__ == '__main__':
    main()
