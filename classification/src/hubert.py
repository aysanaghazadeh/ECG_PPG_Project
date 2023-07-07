from transformers import AutoFeatureExtractor, HubertForSequenceClassification
import torch.nn as nn
from torch.nn import Linear


class Hubert(nn.Module):
    def __init__(self, config):
        super(Hubert, self).__init__()
        self.config = config
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("superb/hubert-base-superb-ks")
        self.model = HubertForSequenceClassification.from_pretrained("superb/hubert-base-superb-ks")
        self.model.config.mask_time_length = 2
        # self.model.config.attention_dropout = 0.01
        # self.model.config.feat_proj_dropout = 0.01
        # self.model.config.activation_dropout = 0.01
        # self.model.config.hidden_dropout = 0.01
        # self.model.config.mask_time_prob = 0.02
        # self.model.config.layerdrop = 0.01
        self.classifier = Linear(self.model.config.num_attention_heads, 1)

    def forward(self, x):
        x = x.squeeze()#.cpu().numpy()
        # x = [x[i] for i in range(len(x))]
        inputs = self.feature_extractor(x, sampling_rate=16000, return_tensors="pt")
        inputs['input_values'] = inputs['input_values'].squeeze().to(device=self.config.device)
        inputs['attention_mask'] = inputs['attention_mask'].squeeze().to(device=self.config.device)
        logits = self.model(**inputs).logits
        output = self.classifier(logits)
        return output.squeeze()

