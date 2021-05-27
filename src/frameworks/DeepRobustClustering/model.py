from torch import nn
from transformers import BertModel


class BertForClassification(nn.Module):
    def __init__(self, path, config, num_labels):
        super(BertForClassification, self).__init__()
        self.bert = BertModel.from_pretrained(
            pretrained_model_name_or_path=path, config=config
        )
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, token_type_ids):

        # outputs: (num_layers, batch_size, max_seq_len, hidden_size)
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Perform mean pooling of token features from last encoder layer
        features = outputs.hidden_states[-1].mean(dim=1)

        # Pass pooled features through classification layer
        logits = self.classifier(features)  # (batch_size, num_labels)
        probabilities = nn.functional.softmax(logits, dim=1)  # (batch_size, num_labels)
        return probabilities, logits