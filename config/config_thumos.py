import numpy as np
import argparse
import os


def parse_args():
    """Parse command line arguments"""
    description = 'Weakly supervised action localization'
    parser = argparse.ArgumentParser(description=description)

    # dataset parameters
    parser.add_argument('--data_path', type=str, default='/data/lizl/THUMOS14', help='Path to dataset')
    parser.add_argument('--exp_name', type=str, required=True, help='Name of the current experiment')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory for results')

    # data parameters
    parser.add_argument('--modal', type=str, default='all', choices=['rgb', 'flow', 'all'], help='Modalities to use')
    parser.add_argument('--num_segments', default=750, type=int, help='Number of segments')
    parser.add_argument('--num_segments1', default=50, type=int, help='Additional segment parameter')
    parser.add_argument('--num_segments2', default=1500, type=int, help='Additional segment parameter')
    parser.add_argument('--scale', default=24, type=int, help='Scale factor for resolution upgrade')
    
    # model parameters
    parser.add_argument('--model_name', required=True, type=str, help='Which model to use')

    # training parameters
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=5000, help='Number of epochs')
    parser.add_argument('--detection_inf_step', default=50, type=int, help='Run detection inference every n steps')
    parser.add_argument('--q_val', default=0.6, type=float, help='q value for GeneralizedCE loss')

    # inference parameters
    parser.add_argument('--inference_only', action='store_true', default=False, help='Run only inference')
    parser.add_argument('--class_th', type=float, default=0.2, help='Class confidence threshold')
    parser.add_argument('--model_file', type=str, default=None, help='Path of pre-trained model file')
    parser.add_argument('--gamma', type=float, default=0.2, help='Gamma for oic class confidence')
    parser.add_argument('--soft_nms', default=False, action='store_true', help='Use soft NMS')
    parser.add_argument('--nms_alpha', default=0.3, type=float, help='Alpha for soft NMS')
    parser.add_argument('--nms_thresh', default=0.4, type=float, help='NMS threshold')
    parser.add_argument('--load_weight', default=False, action='store_true', help='Load pre-trained weights')
    
    # system parameters
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (-1 for no manual seed)')
    parser.add_argument('--verbose', default=False, action='store_true', help='Verbose mode')
    
    return init_args(parser.parse_args())


def init_args(args):
    """Initialize arguments and create necessary directories"""
    args.model_path = os.path.join(args.output_dir, args.exp_name)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    return args


class Config(object):
    """Configuration class for the model and training"""
    def __init__(self, args):
        # Training parameters
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.detection_inf_step = args.detection_inf_step
        self.q_val = args.q_val
        
        # Model parameters
        self.num_classes = 20
        self.modal = args.modal
        self.len_feature = 2048 if self.modal == 'all' else 1024
        self.model_name = args.model_name
        
        # Data parameters
        self.data_path = args.data_path
        self.num_segments = args.num_segments
        self.scale = args.scale
        self.feature_fps = 25
        self.text_features = "/data/lizl/THUMOS14/text_label/texts_THUMOS4_v2.pt"
        # self.text_features = "/data/lizl/THUMOS14/text_label/texts_THUMOS4_CLIP.pt"
        
        # Inference parameters
        self.inference_only = args.inference_only
        self.class_thresh = args.class_th
        self.act_thresh = np.arange(0.1, 1.0, 0.1)
        self.gamma = args.gamma
        self.soft_nms = args.soft_nms
        self.nms_alpha = args.nms_alpha
        self.nms_thresh = args.nms_thresh
        self.load_weight = args.load_weight
        
        # System parameters
        self.output_dir = args.output_dir
        self.model_path = os.path.join(args.output_dir, args.exp_name)
        self.num_workers = args.num_workers
        self.seed = args.seed
        self.verbose = args.verbose
        
        # Paths
        self.gt_path = os.path.join(self.data_path, 'gt.json')
        self.model_file = args.model_file
        
        # Loss weights
        self.lambda1 = 300  # KL loss weight
        self.lambda2 = 0.1   # VL loss weight
        self.lambda3 = 0.5   # Snip loss weight
        
        # Thresholds
        self.alpha_h = 0.9   # High threshold for positive examples
        self.alpha_l = 0.25  # Low threshold for negative examples


class_dict = {
    0: 'BaseballPitch',
    1: 'BasketballDunk',
    2: 'Billiards',
    3: 'CleanAndJerk',
    4: 'CliffDiving',
    5: 'CricketBowling',
    6: 'CricketShot',
    7: 'Diving',
    8: 'FrisbeeCatch',
    9: 'GolfSwing',
    10: 'HammerThrow',
    11: 'HighJump',
    12: 'JavelinThrow',
    13: 'LongJump',
    14: 'PoleVault',
    15: 'Shotput',
    16: 'SoccerPenalty',
    17: 'TennisSwing',
    18: 'ThrowDiscus',
    19: 'VolleyballSpiking'}
