# model/model.py

import torch
import torch.nn as nn
import timm


class SkinClassifier(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super(SkinClassifier, self).__init__()

        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )

        num_features = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output
