"""
models/detector_v2.py — Five-branch deepfake detector for deployment
Branch A: EfficientNet-B3 on raw RGB (ImageNet pretrained)
Branch B: EfficientNet-B0 on spectral diff image (from scratch)
Branch C: MLP on CT handcrafted features (9369-dim)
Branch D: MLP on PRNU noise residual (30-dim)
Branch E: MLP on rPPG features (32-dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class EfficientNetBranch(nn.Module):
    def __init__(self, model_name='efficientnet_b0', pretrained=True, dropout=0.4):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained,
            num_classes=0, global_pool='avg')
        self.dropout = nn.Dropout(p=dropout)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            self.output_dim = self.backbone(dummy).shape[1]

    def forward(self, x):
        return self.dropout(self.backbone(x))


class HandcraftedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropouts, output_dim=256):
        super().__init__()
        self.output_dim = output_dim
        self.input_bn   = nn.BatchNorm1d(input_dim)
        layers, d = [], input_dim
        for h, dr in zip(hidden_dims, dropouts):
            layers += [nn.Linear(d,h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dr)]
            d = h
        layers += [nn.Linear(d, output_dim), nn.BatchNorm1d(output_dim), nn.GELU()]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(self.input_bn(x))


class FiveBranchFusion(nn.Module):
    def __init__(self, dims, fusion_dim=512, dropout=0.3):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Linear(d, fusion_dim),
                          nn.LayerNorm(fusion_dim), nn.GELU())
            for d in dims])
        n = len(dims)
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim*n, 256), nn.ReLU(), nn.Linear(256, n))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim//2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(fusion_dim//2, 1))

    def forward(self, features, return_gates=False):
        projected = [proj(f) for proj,f in zip(self.projs, features)]
        gates  = F.softmax(self.gate(torch.cat(projected, dim=1)), dim=1)
        fused  = sum(gates[:,i:i+1]*p for i,p in enumerate(projected))
        logits = self.classifier(fused).squeeze(1)
        return (logits, gates) if return_gates else logits


class DeepfakeDetectorV2(nn.Module):
    def __init__(self, raw_model='efficientnet_b3', spec_model='efficientnet_b0',
                 mlp_input_dim=9369, prnu_dim=30, rppg_dim=32,
                 fusion_dim=512, fusion_dropout=0.3,
                 efficientnet_dropout=0.4, mlp_hidden_dims=None, mlp_dropouts=None):
        super().__init__()
        if mlp_hidden_dims is None: mlp_hidden_dims = [512, 256]
        if mlp_dropouts    is None: mlp_dropouts    = [0.5, 0.4]
        self.branch_raw  = EfficientNetBranch(raw_model,  True,  efficientnet_dropout)
        self.branch_spec = EfficientNetBranch(spec_model, False, efficientnet_dropout)
        self.branch_hc   = HandcraftedMLP(mlp_input_dim, mlp_hidden_dims, mlp_dropouts, 256)
        self.branch_prnu = HandcraftedMLP(prnu_dim, [64,64], [0.3,0.2], 64)
        self.branch_rppg = HandcraftedMLP(rppg_dim, [64,64], [0.3,0.2], 64)
        self.fusion = FiveBranchFusion(
            dims=[self.branch_raw.output_dim, self.branch_spec.output_dim,
                  self.branch_hc.output_dim,  self.branch_prnu.output_dim,
                  self.branch_rppg.output_dim],
            fusion_dim=fusion_dim, dropout=fusion_dropout)

    def forward(self, raw_rgb, diff_image, handcrafted, prnu, rppg, return_gates=False):
        fa = self.branch_raw(raw_rgb)
        fb = self.branch_spec(diff_image)
        fc = self.branch_hc(handcrafted)
        fd = self.branch_prnu(prnu)
        fe = self.branch_rppg(rppg)
        return self.fusion([fa,fb,fc,fd,fe], return_gates=return_gates)

    def get_optimizer_param_groups(self, lr_backbone=5e-5, lr_heads=5e-4):
        return [
            {'params': self.branch_raw.parameters(),  'lr': lr_backbone},
            {'params': self.branch_spec.parameters(), 'lr': lr_heads},
            {'params': self.branch_hc.parameters(),   'lr': lr_heads},
            {'params': self.branch_prnu.parameters(), 'lr': lr_heads},
            {'params': self.branch_rppg.parameters(), 'lr': lr_heads},
            {'params': self.fusion.parameters(),      'lr': lr_heads},
        ]

    def freeze_raw_backbone(self):
        for p in self.branch_raw.backbone.parameters(): p.requires_grad_(False)

    def unfreeze_raw_backbone(self):
        for p in self.branch_raw.backbone.parameters(): p.requires_grad_(True)

    def count_parameters(self):
        def _c(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {'branch_raw(B3)': _c(self.branch_raw),
                'branch_spec(B0)': _c(self.branch_spec),
                'branch_hc': _c(self.branch_hc),
                'branch_prnu': _c(self.branch_prnu),
                'branch_rppg': _c(self.branch_rppg),
                'fusion': _c(self.fusion), 'total': _c(self)}


def build_detector_v2(config=None):
    if config is None: config = {}
    return DeepfakeDetectorV2(
        raw_model           =config.get('raw_model',            'efficientnet_b3'),
        spec_model          =config.get('spec_model',           'efficientnet_b0'),
        mlp_input_dim       =config.get('mlp_input_dim',        9369),
        prnu_dim            =config.get('prnu_dim',             30),
        rppg_dim            =config.get('rppg_dim',             32),
        fusion_dim          =config.get('fusion_dim',           512),
        fusion_dropout      =config.get('fusion_dropout',       0.3),
        efficientnet_dropout=config.get('efficientnet_dropout', 0.4),
        mlp_hidden_dims     =config.get('mlp_hidden_dims',      [512, 256]),
        mlp_dropouts        =config.get('mlp_dropouts',         [0.5, 0.4]),
    )