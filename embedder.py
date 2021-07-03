import shelve
from typing import List, Tuple, Iterable

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

    def _embed_noncached(self, texts: List[str]) -> Iterable[np.ndarray]:
        if not texts:
            return []

        embeddings, mask, idf = get_bert_embedding(texts, self.scorer._model, self.scorer._tokenizer,
                                                   get_idf_dict(texts, self.scorer._tokenizer),
                                                   device=self.scorer.device)

        for text, embedding in zip(texts, embeddings.cpu().numpy()):
            self.db[text] = embedding
            yield embedding

    def _embed_cached(self, texts: List[str]) -> Iterable[np.ndarray]:
        for text_number, text in enumerate(texts):
            embedding = self.db[text]
            yield embedding

    def tokenize_embed(self, texts: List[str]) -> Tuple[List[List[str]], List[np.ndarray]]:
        if not texts:
            raise ValueError('Cannot tokenize and embed an empty list of texts')

        cached_text_numbers, noncached_text_numbers = set(), set()
        cached_texts, noncached_texts = [], []
        for text_number, text in enumerate(texts):
            if text in self.db:
                cached_text_numbers.add(text_number)
                cached_texts.append(text)
            else:
                noncached_text_numbers.add(text_number)
                noncached_texts.append(text)

        cached_embeddings = iter(self._embed_cached(cached_texts))
        noncached_embeddings = iter(self._embed_noncached(noncached_texts))

        texts_tokens = self._tokenize(texts)
        assert len(texts_tokens) == len(texts)

        embeddings = []
        for text_number, (text, text_tokens) in enumerate(zip(texts, texts_tokens)):
            if text_number in cached_text_numbers:
                embedding = next(cached_embeddings)
            else:
                embedding = next(noncached_embeddings)
            if embedding.shape[0] != len(text_tokens):
                raise ValueError(f'Expected {len(text_tokens)} tokens for "{text}", but received {embedding.shape[0]}')
            embeddings.append(embedding)
        assert len(embeddings) == len(texts)

        return (texts_tokens, embeddings)
