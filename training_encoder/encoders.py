from transformers import AutoTokenizer, AutoModel, ModernBertModel
import torch.nn as nn

class ModernBertEncoder(nn.Module):

    def __init__(self, device="cuda"):
        super(ModernBertEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        self.model = ModernBertModel.from_pretrained(
            "answerdotai/ModernBERT-base"
        )
        self.model.to(device)

    def forward(self, texts):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=True,
            return_tensors="pt",
        ).to("cuda")
        outputs = self.model(**inputs, output_hidden_states=True)

        last_hidden_states = outputs.last_hidden_state
        cls_embedding = last_hidden_states[:, 0, :]

        return cls_embedding

class BertEncoder(nn.Module):

    def __init__(self, device="cuda"):
        super(BertEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
        self.model = AutoModel.from_pretrained(
            "google-bert/bert-base-cased"
        )
        self.model.to(device)

    def forward(self, texts):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=True,
            return_tensors="pt",
        ).to("cuda")
        outputs = self.model(**inputs, output_hidden_states=True)

        last_hidden_states = outputs.last_hidden_state
        cls_embedding = last_hidden_states[:, 0, :]

        return cls_embedding

class AllMiniLMEncoder(nn.Module):

    def __init__(self, device="cuda"):
        super(AllMiniLMEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model.to(device)

    def forward(self, texts):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=True,
            return_tensors="pt",
        ).to("cuda")
        outputs = self.model(**inputs, output_hidden_states=True)

        last_hidden_states = outputs.last_hidden_state
        cls_embedding = last_hidden_states[:, 0, :]

        return cls_embedding
