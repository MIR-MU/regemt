import shelve
from typing import List, Tuple

import numpy as np
from bert_score import BERTScorer
from bert_score.utils import get_bert_embedding, get_idf_dict


class ContextualEmbedder:
    vector_size = 1024
    db_filename = 'bert-embeddings-db'

    def __init__(self, lang: str):
        self.scorer = BERTScorer(lang=lang)
        self.db = shelve.open(self.db_filename)

    def _tokenize(self, texts: List[str]) -> List[List[str]]:
        input_ids_batch = self.scorer._tokenizer(texts).input_ids
        token_batch = [[self.scorer._tokenizer.decode(input_id) for input_id in input_ids]
                       for input_ids in input_ids_batch]
        return token_batch

    def _embed_noncached(self, texts: List[str]) -> np.ndarray:
        embeddings, mask, idf = get_bert_embedding(texts, self.scorer._model, self.scorer._tokenizer,
                                                   get_idf_dict(texts, self.scorer._tokenizer),
                                                   device=self.scorer.device)
        embeddings = embeddings.cpu().numpy()
        for text, embedding in zip(texts, embeddings):
            self.db[text] = embedding
        return embeddings

    def _embed_cached(self, texts: List[str]) -> np.ndarray:
        embeddings = np.empty(shape=(len(texts), self.vector_size))
        for text_number, text in enumerate(texts):
            embedding = self.db[text]
            embeddings[text_number, :] = embedding
        return embeddings

    def tokenize_embed(self, texts: List[str]) -> Tuple[List[List[str]], np.ndarray]:
        if not texts:
            raise ValueError('Cannot tokenize and embed an empty list of texts')

        cached_text_numbers, noncached_text_numbers = [], []
        cached_texts, noncached_texts = [], []
        for text_number, text in enumerate(texts):
            if text in self.db:
                cached_text_numbers.append(text_number)
                cached_texts.append(text)
            else:
                noncached_text_numbers.append(text_number)
                noncached_texts.append(text)

        if not noncached_texts:
            embeddings = self._embed_cached(cached_texts)
        elif not cached_texts:
            embeddings = self._embed_noncached(noncached_texts)
        else:
            embeddings = np.empty(shape=(len(texts), self.vector_size))
            embeddings[cached_text_numbers, :] = self._embed_cached(cached_texts)
            embeddings[noncached_text_numbers, :] = self._embed_noncached(noncached_texts)

        input_ids_batch = self._tokenize(texts)
        return (input_ids_batch, embeddings)
