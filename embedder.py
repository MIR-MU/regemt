import shelve
from typing import List, Tuple, Iterable, Dict, Any
from urllib.request import urlretrieve
from pathlib import Path

import torch
from tqdm.autonotebook import tqdm
import numpy as np
from bert_score import BERTScorer
from transformers import BatchEncoding
from gensim.models.keyedvectors import KeyedVectors

Text = str
Tokens = List[str]
Language = str
Embeddings = np.ndarray


class FastTextEmbedder:
    all_keyedvectors: Dict[Language, KeyedVectors] = dict()

    def __init__(self, lang: str):
        self.lang = lang

    @property
    def keyedvectors(self, base_path: Path = Path('embeddings')) -> KeyedVectors:
        if self.lang not in self.all_keyedvectors:
            path = base_path / f'cc.{self.lang}.300.vec.gz'
            if not path.exists():
                print(f'Downloading fastText embeddings for language {self.lang}')
                url = f'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{self.lang}.300.vec.gz'
                base_path.mkdir(parents=False, exist_ok=True)
                urlretrieve(url, path)
            print(f'Loading fastText embeddings for language {self.lang}')
            keyedvectors = KeyedVectors.load_word2vec_format(path)
            self.all_keyedvectors[self.lang] = keyedvectors
        return self.all_keyedvectors[self.lang]

    def __hash__(self) -> int:
        return hash(self.lang)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ContextualEmbedder):
            return NotImplemented
        return all([
            self.lang == other.lang,
        ])


class ContextualEmbedder:
    vector_size: int
    gpus = [0]  # TODO
    batch_size = 10
    device = "cuda" if torch.cuda.device_count() > 0 else "cpu"

    diskcaches: Dict[Language, shelve.Shelf] = dict()
    ramcaches: Dict[Language, Dict[Text, Tuple[Tokens, Embeddings]]] = dict()
    scorers: Dict[Language, BERTScorer] = dict()

    def __init__(self, lang: str, use_diskcache: bool = True, use_ramcache: bool = False, reference_free: bool = False):
        self.lang = lang if not reference_free else "multi"  # lang other than en and zh retrieve multilingual model
        self.vector_size = self.scorer._model.config.hidden_size
        self.use_diskcache = use_diskcache
        self.use_ramcache = use_ramcache

    @property
    def diskcache(self) -> shelve.Shelf:
        assert self.use_diskcache
        if self.lang not in self.diskcaches:
            self.diskcaches[self.lang] = shelve.open(f'embedder-diskcache-{self.lang}')
        return self.diskcaches[self.lang]

    @property
    def ramcache(self) -> Dict[Text, Tuple[Tokens, Embeddings]]:
        assert self.use_ramcache
        if self.lang not in self.ramcaches:
            self.ramcaches[self.lang] = dict()
        return self.ramcaches[self.lang]

    @property
    def scorer(self) -> BERTScorer:
        if self.lang not in self.scorers:
            print(f'Loading BERTScorer for language {self.lang}')
            self.scorers[self.lang] = BERTScorer(lang=self.lang)
        return self.scorers[self.lang]

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

                if self.use_diskcache:
                    self.diskcache[text] = (tokens_nopad, embeddings_nopad)
                if self.use_ramcache:
                    self.ramcache[text] = (tokens_nopad, embeddings_nopad)
                yield tokens_nopad, embeddings_nopad

    def _embed_cached(self, texts: List[Text]) -> Iterable[Tuple[Tokens, Embeddings]]:
        assert self.use_diskcache
        for text in texts:
            if self.use_ramcache and text in self.ramcache:
                tokens, embeddings = self.ramcache[text]
            else:
                tokens, embeddings = self.diskcache[text]
                if self.use_ramcache:
                    self.ramcache[text] = (tokens, embeddings)
            assert embeddings.shape[0] == len(tokens)
            yield tokens, embeddings

    def tokenize_embed(self, texts: List[Text]) -> Tuple[List[Tokens], List[Embeddings]]:
        if not texts:
            raise ValueError('Cannot tokenize and embed an empty list of texts')

        cached_text_numbers, noncached_text_numbers = set(), set()
        cached_texts, noncached_texts = [], []
        for text_number, text in enumerate(texts):
            if self.use_diskcache and text in self.diskcache:
                cached_text_numbers.add(text_number)
                cached_texts.append(text)
            else:
                noncached_text_numbers.add(text_number)
                noncached_texts.append(text)

        cached_embeddings = iter(self._embed_cached(cached_texts))
        noncached_embeddings = iter(self._embed_noncached(noncached_texts))

        if self.use_diskcache and noncached_texts:
            self.diskcache.sync()

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

    def __hash__(self) -> int:
        return hash((self.lang, self.vector_size))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ContextualEmbedder):
            return NotImplemented
        return all([
            self.lang == other.lang,
            self.vector_size == other.vector_size,
        ])
