from typing import List
from itertools import chain

from gensim.models.fasttext import load_facebook_vectors
from gensim.models.keyedvectors import KeyedVectors, _add_word_to_kv
from gensim.corpora import Dictionary
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

from common import Metric, Judgements, AugmentedCorpus
from embedder import ContextualEmbedder


class ContextualWMD(Metric):
    label = "WMD_contextual"
    w2v_model = None
    embedder = None
    stopwords = None
    zipped_test_corpus = None

    def __init__(self, tgt_lang: str):
        self.embedder = ContextualEmbedder(lang=tgt_lang)
        if tgt_lang == "en":
            nltk.download('stopwords')
            self.stopwords = stopwords.words('english')
        else:
            raise ValueError(tgt_lang)

    def fit(self, train_judgements: Judgements, test_judgements: Judgements):
        test_ref_corpus, test_ref_embs = self.embedder.tokenize_embed([t[0] for t in test_judgements.references])
        test_trans_corpus, test_trans_embs = self.embedder.tokenize_embed(test_judgements.translations)

        augmented_test_reference_corpus = AugmentedCorpus('test-reference', test_ref_corpus)
        augmented_test_translation_corpus = AugmentedCorpus('test-translation', test_trans_corpus)

        self.zipped_test_corpus = list(zip(
            augmented_test_reference_corpus.corpus,
            augmented_test_translation_corpus.corpus,
        ))
        corpus = augmented_test_reference_corpus.corpus + augmented_test_translation_corpus.corpus
        # different dims, we can not use np (can be aligned in precedence)
        corpus_embeddings = chain(test_ref_embs, test_trans_embs)

        # We only use words from test corpus, since we don't care about words from train corpus
        self.dictionary = Dictionary(corpus, prune_at=None)

        self.w2v_model = KeyedVectors(self.embedder.vector_size, len(self.dictionary), dtype=float)
        for augmented_tokens, tokens_embeddings in tqdm(zip(corpus, corpus_embeddings),
                                                        desc=f'{self.label}: construct contextual embeddings',
                                                        total=len(corpus)):
            for token_index, (token, token_embedding) in enumerate(zip(augmented_tokens, tokens_embeddings)):
                _add_word_to_kv(self.w2v_model, None, token, token_embedding, len(self.dictionary))

    def compute(self, judgements: Judgements) -> List[float]:
        out_scores = [self.w2v_model.wmdistance(reference_words, translation_words)
                      for reference_words, translation_words
                      in tqdm(self.zipped_test_corpus, desc=self.label)]
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
        out_scores = [self.w2v_model.wmdistance(reference_words, translation_words)
                      for reference_words, translation_words
                      in judgements.get_tokenized_texts(self.stopwords, desc=self.label)]
        return out_scores
