import torch.nn as nn


class BaseHeadSplit(nn.Module):
    def __init__(self, base, head, **kwargs):
        super(BaseHeadSplit, self).__init__()

        self.base = base
        self.head = head

        for k, v in kwargs.items():
            setattr(self, k, v)
        
    def forward(self, x):
        out = self.base(x)
        out = self.head(out)

        return out
    
class TwoHead(nn.Module):
    def __init__(self, base, **kwargs):
        super(TwoHead, self).__init__()

        self.base = base
        fet_dim = base.fc.in_features
        num_class = base.fc.out_features
        self.base.fc = nn.Identity()
        self.fc = nn.Linear(fet_dim, num_class)

        self.fc_aux = nn.Sequential(
            nn.Linear(fet_dim, fet_dim // 4),
            nn.BatchNorm1d(fet_dim // 4, track_running_stats=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            
            nn.Linear(fet_dim // 4, fet_dim // 4),
            nn.BatchNorm1d(fet_dim // 4, track_running_stats=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            
            nn.Linear(fet_dim // 4, num_class)
        )

    def get_fet(self, x):
        return self.base(x)
    
    def forward(self, x):
        fet = self.base(x)
        logits = self.fc(fet)
        if self.training:
            logits_aux = self.fc_aux(fet)
            return logits, logits_aux

        return logits
