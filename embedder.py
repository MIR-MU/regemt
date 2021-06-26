from typing import List, Tuple

import numpy as np
from bert_score import BERTScorer
from bert_score.utils import get_bert_embedding, get_idf_dict


class ContextualEmbedder:
    vector_size = 1024

    def __init__(self, lang: str):
        self.scorer = BERTScorer(lang=lang)

    def tokenize_embed(self, texts: List[str]) -> Tuple[List[List[str]], np.ndarray]:
        input_ids_batch = self.scorer._tokenizer(texts).input_ids
        token_batch = [[self.scorer._tokenizer.decode(input_id) for input_id in input_ids]
                       for input_ids in input_ids_batch]

        embeddings, mask, idf = get_bert_embedding(texts, self.scorer._model, self.scorer._tokenizer,
                                                   get_idf_dict(texts, self.scorer._tokenizer),
                                                   device=self.scorer.device)
        return token_batch, embeddings.cpu().numpy()
