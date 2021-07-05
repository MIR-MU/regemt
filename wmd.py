from collections import defaultdict
from multiprocessing import Pool
from typing import List, Iterable, Tuple, Optional

from gensim.models.fasttext import load_facebook_vectors
from gensim.models.keyedvectors import KeyedVectors, _add_word_to_kv
from gensim.corpora import Dictionary
import nltk
from nltk.corpus import stopwords
import numpy as np
from tqdm.autonotebook import tqdm

from common import Metric, Judgements, AugmentedCorpus
from embedder import ContextualEmbedder


WMD_W2V_MODEL: Optional[KeyedVectors] = None


def _get_wmds_worker(args: Tuple[List[str], List[str]]) -> float:
    reference_words, translation_words = args
    distance = WMD_W2V_MODEL.wmdistance(reference_words, translation_words)
    return distance


def get_wmds(w2v_model: KeyedVectors, tokenized_texts: Iterable[Tuple[List[str], List[str]]]) -> List[float]:
    # We abuse global variables to get fast parallel WMD
    global WMD_W2V_MODEL
    WMD_W2V_MODEL = w2v_model
    distances = []
    with Pool(None) as pool:
        for distance in pool.imap(_get_wmds_worker, tokenized_texts):
            distances.append(distance)
    WMD_W2V_MODEL = None
    return distances


class ContextualWMD(Metric):
    label = "WMD_contextual"
    w2v_model = None
    embedder = None
    test_judgements = None
    zipped_test_corpus = None

    def __init__(self, tgt_lang: str):
        self.embedder = ContextualEmbedder(lang=tgt_lang)
        if tgt_lang != "en":
            raise ValueError(tgt_lang)

    def fit(self, train_judgements: Judgements, test_judgements: Judgements):
        self.test_judgements = test_judgements

        test_ref_corpus, test_ref_embs = self.embedder.tokenize_embed([t[0] for t in test_judgements.references])
        test_trans_corpus, test_trans_embs = self.embedder.tokenize_embed(test_judgements.translations)

        augmented_test_reference_corpus = AugmentedCorpus('test-reference', test_ref_corpus)
        augmented_test_translation_corpus = AugmentedCorpus('test-translation', test_trans_corpus)

        self.zipped_test_corpus = list(zip(
            augmented_test_reference_corpus.corpus,
            augmented_test_translation_corpus.corpus,
        ))

        # We only use words from test corpus, since we don't care about words from train corpus
        corpus = augmented_test_reference_corpus.corpus + augmented_test_translation_corpus.corpus
        embeddings = test_ref_embs + test_trans_embs
        dictionary = Dictionary(corpus, prune_at=None)

        self.w2v_model = KeyedVectors(self.embedder.vector_size, len(dictionary), dtype=float)
        for augmented_tokens, tokens_embeddings in tqdm(zip(corpus, embeddings),
                                                        desc=f'{self.label}: construct contextual embeddings',
                                                        total=len(corpus)):
            for token, token_embedding in zip(augmented_tokens, tokens_embeddings):
                _add_word_to_kv(self.w2v_model, None, token, token_embedding, len(dictionary))

    def compute(self, judgements: Judgements) -> List[float]:
        if judgements != self.test_judgements:
            raise ValueError('Tne judgements are different from the test_judgements used in fit()')

        out_scores = get_wmds(self.w2v_model, tqdm(self.zipped_test_corpus, desc=self.label))
        return out_scores


class DecontextualizedWMD(Metric):

    label = "WMD_decontextualized"
    w2v_model = None
    test_judgements = None
    zipped_test_corpus = None

    def __init__(self, tgt_lang: str):
        self.embedder = ContextualEmbedder(lang=tgt_lang)
        if tgt_lang != "en":
            raise ValueError(tgt_lang)

    def fit(self, train_judgements: Judgements, test_judgements: Judgements):
        self.test_judgements = test_judgements

        test_ref_corpus, test_ref_embs = self.embedder.tokenize_embed([t[0] for t in test_judgements.references])
        test_trans_corpus, test_trans_embs = self.embedder.tokenize_embed(test_judgements.translations)

        self.zipped_test_corpus = list(zip(test_ref_corpus, test_trans_corpus))

        # We only use words from test corpus, since we don't care about words from train corpus
        corpus = test_ref_corpus + test_trans_corpus
        embeddings = test_ref_embs + test_trans_embs

        # We average embeddings for all occurences for a term
        decontextualized_embeddings = defaultdict(lambda: [])
        for tokens, tokens_embeddings in zip(corpus, embeddings):
            for token, token_embedding in zip(tokens, tokens_embeddings):
                decontextualized_embeddings[token].append(token_embedding)

        self.w2v_model = KeyedVectors(self.embedder.vector_size, len(decontextualized_embeddings), dtype=float)
        for token, token_embeddings in tqdm(decontextualized_embeddings.items(),
                                            f'{self.label}: construct decontextualized embeddings'):
            token_embedding = np.mean(token_embeddings, axis=0)
            _add_word_to_kv(self.w2v_model, None, token, token_embedding, len(decontextualized_embeddings))

    def compute(self, judgements: Judgements) -> List[float]:
        if judgements != self.test_judgements:
            raise ValueError('Tne judgements are different from the test_judgements used in fit()')

        out_scores = get_wmds(self.w2v_model, tqdm(self.zipped_test_corpus, desc=self.label))
        return out_scores


class WMD(Metric):

    label = "WMD"
    w2v_model = None
    stopwords = None

    def __init__(self, tgt_lang: str):
        if tgt_lang == "en":
            self.w2v_model = load_facebook_vectors('embeddings/cc.en.300.bin')
            self.w2v_model.init_sims(replace=True)
            nltk.download('stopwords')
            self.stopwords = stopwords.words('english')

        else:
            raise ValueError(tgt_lang)

    def fit(self, train_judgements: Judgements, test_judgements: Judgements):
        pass

    def compute(self, judgements: Judgements) -> List[float]:
        out_scores = get_wmds(self.w2v_model, judgements.get_tokenized_texts(self.stopwords, desc=self.label))
        return out_scores
