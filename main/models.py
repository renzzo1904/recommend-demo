## File to develop the models that create the recommendations ##

import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class ModelClass:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
        self.embeddings_model = AutoModel.from_pretrained(
            "intfloat/multilingual-e5-base"
        )
        self.clusterer = DBSCAN(eps=0.5, min_samples=5)

    def create_embeddings(self, text: str = "Hello World!") -> np.array:
        """
        Create embeddings for a list of strings using Mutilingual E5 .

        Args:
            text (str) : string to be embedded .

        Returns:
            np.ndarray: Embeddings matrix.
        """
        # Tokenize the input texts
        batch_dict = self.tokenizer(
            "query:" + text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        outputs = self.embeddings_model(**batch_dict)
        embeddings = average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )

        return F.normalize(embeddings, p=2, dim=1).detach().numpy()

    def perform_clustering(self, embeddings) -> list:
        """
        Perform clustering on the embeddings using KMeans algorithm.

        Args:
            embeddings (np.ndarray): Embeddings matrix.

        Returns:
            np.ndarray: Cluster labels.
        """
        scaler = StandardScaler()
        normalized_embeddings = scaler.fit_transform(embeddings)
        cluster_labels = self.clusterer.fit_predict(normalized_embeddings)
        return cluster_labels
