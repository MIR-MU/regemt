from collections import defaultdict
from typing import List

from gensim.models import TfidfModel
from gensim.models.fasttext import load_facebook_vectors
from gensim.models.keyedvectors import KeyedVectors, _add_word_to_kv
from gensim.corpora import Dictionary
import nltk
from nltk.corpus import stopwords
import numpy as np
from tqdm.autonotebook import tqdm

from common import Metric, ReferenceFreeMetric, Judgements, AugmentedCorpus
from embedder import ContextualEmbedder
from _wmd import get_wmds, get_wmds_tfidf


class ContextualWMD(ReferenceFreeMetric):

    label = "WMD_contextual"

    def __init__(self, tgt_lang: str, reference_free: bool = False):
        self.embedder = ContextualEmbedder(lang=tgt_lang, reference_free=reference_free)
        self.reference_free = reference_free

    def compute(self, judgements: Judgements) -> List[float]:
        if self.reference_free:
            ref_corpus, ref_embs = self.embedder.tokenize_embed([t for t in judgements.src_texts])
        else:
            ref_corpus, ref_embs = self.embedder.tokenize_embed([t[0] for t in judgements.references])
        trans_corpus, trans_embs = self.embedder.tokenize_embed(judgements.translations)

        augmented_reference_corpus = AugmentedCorpus('test-reference', ref_corpus)
        augmented_translation_corpus = AugmentedCorpus('test-translation', trans_corpus)

        corpus = augmented_reference_corpus.corpus + augmented_translation_corpus.corpus
        embeddings = ref_embs + trans_embs
        dictionary = Dictionary(corpus, prune_at=None)

        w2v_model = KeyedVectors(self.embedder.vector_size, len(dictionary), dtype=float)
        for augmented_tokens, tokens_embeddings in tqdm(zip(corpus, embeddings),
                                                        desc=f'{self.label}: construct contextual embeddings',
                                                        total=len(corpus)):
            for token, token_embedding in zip(augmented_tokens, tokens_embeddings):
                _add_word_to_kv(w2v_model, None, token, token_embedding, len(dictionary))

        zipped_corpus = list(zip(augmented_reference_corpus.corpus, augmented_translation_corpus.corpus))
        out_scores = get_wmds(w2v_model, tqdm(zipped_corpus, desc=self.label))
        return out_scores

    def compute_ref_free(self, judgements: Judgements) -> List[float]:
        return self.compute(judgements)


class DecontextualizedWMD(ReferenceFreeMetric):

    label = "WMD_decontextualized"

    def __init__(self, tgt_lang: str, use_tfidf: bool, reference_free: bool = False):
        self.embedder = ContextualEmbedder(lang=tgt_lang, reference_free=reference_free)
        self.reference_free = reference_free

        self.use_tfidf = use_tfidf
        if use_tfidf:
            self.label = self.label + "_tfidf"

    def compute(self, judgements: Judgements) -> List[float]:
        if self.reference_free:
            ref_corpus, ref_embs = self.embedder.tokenize_embed([t for t in judgements.src_texts])
        else:
            ref_corpus, ref_embs = self.embedder.tokenize_embed([t[0] for t in judgements.references])
        trans_corpus, trans_embs = self.embedder.tokenize_embed(judgements.translations)

        corpus = ref_corpus + trans_corpus
        embeddings = ref_embs + trans_embs

        if self.use_tfidf:
            dictionary = Dictionary(corpus)
            tfidf = TfidfModel(dictionary=dictionary, smartirs='nfx')

        # We average embeddings for all occurences for a term to get "decontextualized" embeddings
        decontextualized_embeddings = defaultdict(lambda: [])
        for tokens, tokens_embeddings in zip(corpus, embeddings):
            for token, token_embedding in zip(tokens, tokens_embeddings):
                decontextualized_embeddings[token].append(token_embedding)

        w2v_model = KeyedVectors(self.embedder.vector_size, len(decontextualized_embeddings), dtype=float)
        for token, token_embeddings in tqdm(decontextualized_embeddings.items(),
                                            f'{self.label}: construct decontextualized embeddings'):
            token_embedding = np.mean(token_embeddings, axis=0)
            _add_word_to_kv(w2v_model, None, token, token_embedding, len(decontextualized_embeddings))

        zipped_corpus = list(zip(ref_corpus, trans_corpus))
        tokenized_texts = tqdm(zipped_corpus, desc=self.label)
        if self.use_tfidf:
            out_scores = get_wmds_tfidf(w2v_model, dictionary, tfidf, tokenized_texts)
        else:
            out_scores = get_wmds(w2v_model, tokenized_texts)
        return out_scores

    def compute_ref_free(self, judgements: Judgements) -> List[float]:
        return self.compute(judgements)


class WMD(Metric):

    label = "WMD"
    w2v_model = None

    def __init__(self, tgt_lang: str, use_tfidf: bool):
        if tgt_lang == "en":
            self.w2v_model = load_facebook_vectors('embeddings/cc.en.300.bin')
            nltk.download('stopwords')
            self.stopwords = stopwords.words('english')
        else:
            raise ValueError(tgt_lang)

        self.use_tfidf = use_tfidf
        if use_tfidf:
            self.label = self.label + "_tfidf"

    def compute(self, judgements: Judgements) -> List[float]:
        ref_corpus, trans_corpus = map(
            list, zip(*judgements.get_tokenized_texts(self.stopwords, desc=self.label)))

        corpus = ref_corpus + trans_corpus

        if self.use_tfidf:
            dictionary = Dictionary(corpus)
            tfidf = TfidfModel(dictionary=dictionary, smartirs='nfx')

        tokenized_texts = judgements.get_tokenized_texts(self.stopwords, desc=self.label)
        if self.use_tfidf:
            out_scores = get_wmds_tfidf(self.w2v_model, dictionary, tfidf, tokenized_texts)
        else:
            out_scores = get_wmds(self.w2v_model, tokenized_texts)
        return out_scores
