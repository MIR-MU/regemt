import shelve
from typing import List, Tuple, Iterable

import torch
from tqdm import tqdm

import numpy as np
from bert_score import BERTScorer
from transformers import BatchEncoding

Text = str
Tokens = List[str]
Embeddings = np.ndarray


class ContextualEmbedder:
    vector_size = 1024
    gpus = [0]  # TODO
    batch_size = 10
    device = "cuda" if torch.cuda.device_count() > 0 else "cpu"

    db = shelve.open('bert-embeddings-db')

    def __init__(self, lang: str, use_db: bool = True):
        self.scorer = BERTScorer(lang=lang)
        self.use_db = True

    def _get_bert_embeddings_parallel(self, inputs_batch: BatchEncoding) -> Embeddings:
        with torch.no_grad():
            batch_output = self.scorer._model(**inputs_batch)
            embeddings = batch_output[0].detach().cpu().numpy()

            return embeddings

    def _embed_noncached(self, texts: List[Text]) -> Iterable[Tuple[Tokens, Embeddings]]:
        if not texts:
            return []

        batch_iter = range(0, len(texts), self.batch_size)
        for i, j in tqdm(((batch_i, batch_i + self.batch_size) for batch_i in batch_iter),
                         desc="BERT embeddings", total=int(len(texts) / self.batch_size)):

            texts_batch = texts[i: j]

            inputs_batch = self.scorer._tokenizer(texts_batch, return_tensors="pt",
                                                  padding=True, truncation=True).to(self.device)

            tokens_batch = [[self.scorer._tokenizer.decode(input_id) for input_id in input_ids]
                            for input_ids in inputs_batch['input_ids']]

            embeddings_batch = self._get_bert_embeddings_parallel(inputs_batch)

            for text, tokens, embeddings in zip(texts_batch, tokens_batch, embeddings_batch):

                embeddings_nopad = np.array([e for t, e in zip(tokens, embeddings)
                                            if t not in self.scorer._tokenizer.all_special_tokens])
                tokens_nopad = [t for t in tokens if t not in self.scorer._tokenizer.all_special_tokens]

                assert len(tokens_nopad) == embeddings_nopad.shape[0], \
                    "'%s': num_tokens: %s" % (tokens_nopad, embeddings_nopad.shape[0])

                if self.use_db:
                    self.db[text] = (tokens_nopad, embeddings_nopad)
                yield tokens_nopad, embeddings_nopad

    def _embed_cached(self, texts: List[Text]) -> Iterable[Tuple[Tokens, Embeddings]]:
        assert self.use_db
        for text in texts:
            tokens, embeddings = self.db[text]
            assert embeddings.shape[0] == len(tokens)
            yield tokens, embeddings

    def tokenize_embed(self, texts: List[Text]) -> Tuple[List[Tokens], List[Embeddings]]:
        if not texts:
            raise ValueError('Cannot tokenize and embed an empty list of texts')

        cached_text_numbers, noncached_text_numbers = set(), set()
        cached_texts, noncached_texts = [], []
        for text_number, text in enumerate(texts):
            if self.use_db and text in self.db:
                cached_text_numbers.add(text_number)
                cached_texts.append(text)
            else:
                noncached_text_numbers.add(text_number)
                noncached_texts.append(text)

        cached_embeddings = iter(self._embed_cached(cached_texts))
        noncached_embeddings = iter(self._embed_noncached(noncached_texts))

        if self.use_db and noncached_texts:
            self.db.sync()

        texts_embeddings = []
        texts_tokens = []
        for text_number, text in enumerate(texts):
            if text_number in cached_text_numbers:
                tokens, embeddings = next(cached_embeddings)
            else:
                tokens, embeddings = next(noncached_embeddings)
            assert len(embeddings) == len(tokens)
            texts_embeddings.append(embeddings)
            texts_tokens.append(tokens)
        assert len(texts_embeddings) == len(texts)
        assert len(texts_tokens) == len(texts)

        return texts_tokens, texts_embeddings
