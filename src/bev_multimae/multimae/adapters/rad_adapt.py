# Adapter from radar to ecoder
import torch.nn as nn
from bev_multimae.preprocessing.BEV.dynamic_pillar_vfe import PointPillarScatter

class RadarAdapter(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.grid_size = cfg.grid_size
        self.d_model = cfg.d_model
        self.num_point_features = cfg.num_point_features

        self.scatter = PointPillarScatter(grid_size=self.grid_size[:2], num_point_features=6)
        
    def forward(self):
        ...