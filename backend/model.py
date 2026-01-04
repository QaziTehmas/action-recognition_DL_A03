import torch
import torch.nn as nn
from torchvision import models

class SharedResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        except Exception:
            resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC layer
        self.out_dim = 2048
    
    def forward(self, x):
        f = self.backbone(x)  # (N, 2048, 1, 1)
        return f.view(f.size(0), -1)  # (N, 2048)

class ActionHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

class CaptionDecoder(nn.Module):
    def __init__(self, in_dim, vocab_size, embed_dim, hidden_dim, num_layers, pad_id):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.to_h0 = nn.Linear(in_dim, hidden_dim)
        self.to_c0 = nn.Linear(in_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.num_layers = num_layers
    
    def forward(self, img_feat, captions_in):
        emb = self.embed(captions_in)
        h0 = torch.tanh(self.to_h0(img_feat))
        c0 = torch.tanh(self.to_c0(img_feat))
        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = c0.unsqueeze(0).repeat(self.num_layers, 1, 1)
        out_seq, _ = self.lstm(emb, (h0, c0))
        logits = self.out(out_seq)
        return logits

class MultiTaskModel(nn.Module):
    def __init__(self, num_actions, vocab_size, embed_dim, hidden_dim, num_layers, pad_id):
        super().__init__()
        self.cnn = SharedResNet50()
        self.action_head = ActionHead(self.cnn.out_dim, num_actions)
        self.caption_head = CaptionDecoder(self.cnn.out_dim, vocab_size, embed_dim, hidden_dim, num_layers, pad_id)
    
    def forward(self, images, captions_in=None):
        img_feat = self.cnn(images)
        action_logits = self.action_head(img_feat)
        
        caption_logits = None
        if captions_in is not None:
            caption_logits = self.caption_head(img_feat, captions_in)
            
        return action_logits, caption_logits
