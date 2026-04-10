import torch
import torch.nn as nn
import math

def norm(tensor):
    tensor_min = tensor.min(dim=1, keepdim=True)[0]
    tensor_max = tensor.max(dim=1, keepdim=True)[0]
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    return normalized_tensor

class VLR_Net(nn.Module):
    def __init__(self, len_feature, num_classes):
        super(VLR_Net, self).__init__()
        self.len_feature = len_feature
        self.num_classes = num_classes
        
        self.text_proj = nn.Sequential(
            nn.Linear(768, 768),
        )
        self.video_proj = nn.Sequential(
            nn.Linear(768, 768),
        )

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=2048+768, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.cls = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=self.num_classes, kernel_size=1, padding=0)
        )

        # Create action networks using a helper method
        self.action_rgb = self._create_action_network(self.len_feature // 2)
        self.action_flow = self._create_action_network(self.len_feature // 2)
        self.action_vl = self._create_action_network(768)
    
    def _create_action_network(self, input_channels):
        """Create a standard action network with given input channels"""
        return nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, padding=0),
        )

    def forward(self, rf_feat, vl_feat, text_feat):
        rf_feat = rf_feat.permute(0, 2, 1)  # [16,750,2048] -> [16,2048,750]
        flow = rf_feat[:, 1024:, :]
        rgb = rf_feat[:, :1024, :]

        action_flow = torch.sigmoid(self.action_flow(flow))
        action_rgb = torch.sigmoid(self.action_rgb(rgb))
        action_vl = torch.sigmoid(self.action_vl(vl_feat.permute(0, 2, 1)))

        rf_feat = torch.cat([rf_feat, vl_feat.permute(0, 2, 1)], dim=1)
        emb = self.encoder(rf_feat)

        cas = self.cls(emb).permute(0, 2, 1)

        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        vl_feat = vl_feat / vl_feat.norm(dim=-1, keepdim=True)

        vl_feat_prj = (self.video_proj(vl_feat) + vl_feat) / 2
        text_feat_prj = (self.text_proj(text_feat) + text_feat) / 2

        text_feat_prj = text_feat_prj / text_feat_prj.norm(dim=-1, keepdim=True)
        vl_feat_prj = vl_feat_prj / vl_feat_prj.norm(dim=-1, keepdim=True)

        sim_text_prj = torch.einsum("blh,nh->bln", vl_feat, text_feat_prj)
        sim_vl_prj = torch.einsum("blh,nh->bln", vl_feat_prj, text_feat)
        sim_prj = torch.einsum("blh,nh->bln", vl_feat_prj, text_feat_prj)

        text_prj_score = norm(sim_text_prj)
        vl_prj_score = norm(sim_vl_prj)

        sim_matrix = (text_prj_score + vl_prj_score) / 2

        return cas, action_flow, action_rgb, action_vl, sim_matrix, sim_text_prj, sim_vl_prj, sim_prj
