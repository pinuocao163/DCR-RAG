import torch
from transformers import XLMRobertaForSequenceClassification
from transformers.models.xlm_roberta.modeling_xlm_roberta import (
    XLMRobertaClassificationHead,
)


class PaperScoreHead(XLMRobertaClassificationHead):
    def forward(self, features, **kwargs):
        hidden = self.dropout(features[:, 0, :])
        hidden = torch.relu(self.dense(hidden))
        hidden = self.dropout(hidden)
        return self.out_proj(hidden)


class DraftPassageScorer(XLMRobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = PaperScoreHead(config)
        self.classifier.apply(self._init_weights)
