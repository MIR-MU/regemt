from typing import List, Any, Iterable, Tuple
from itertools import product, chain

import numpy as np
from gensim.corpora import Dictionary
from gensim.similarities import WordEmbeddingSimilarityIndex, SparseTermSimilarityMatrix
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from gensim.models.fasttext import load_facebook_vectors
from gensim.models.keyedvectors import KeyedVectors, _add_word_to_kv
import nltk
from tqdm import tqdm
from scipy.sparse import dok_matrix, csr_matrix

from common import Metric, Judgements
from embedder import ContextualEmbedder


class ContextualSCM(Metric):
    label = "SCM_contextual"
    stopwords = None
    similarity_matrix = None
    dictionary = None
    tfidf = None
    embedder = None

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

        def augment_corpus(prefix: Any, corpus: Iterable[List[str]]) -> List[List[str]]:
            return [augment_tokens(prefix, tokens) for tokens_index, tokens in enumerate(corpus)]

        def augment_tokens(prefix: Any, tokens: Iterable[str]) -> List[str]:
            return [" ".join([prefix, str(token_index), token]) for token_index, token in enumerate(tokens)]

        augmented_test_reference_corpus = augment_corpus('test-reference', test_ref_corpus)
        augmented_test_translation_corpus = augment_corpus('test-translation', test_trans_corpus)

        self.zipped_test_corpus = list(zip(augmented_test_reference_corpus, augmented_test_translation_corpus))
        corpus = augmented_test_reference_corpus + augmented_test_translation_corpus
        # different dims, we can not use np (can be aligned in precedence)
        corpus_embeddings = chain(test_ref_embs, test_trans_embs)

        # We only use words from test corpus, since we don't care about words from train corpus
        self.dictionary = Dictionary(corpus)

        embeddings = KeyedVectors(self.embedder.vector_size, len(self.dictionary), dtype=np.float)
        for augmented_tokens, tokens_embeddings in tqdm(zip(corpus, corpus_embeddings),
                                                        desc=f'{self.label}: construct contextual embeddings'):
            for token_index, (token, token_embedding) in enumerate(zip(augmented_tokens, tokens_embeddings)):
                _add_word_to_kv(embeddings, None, token, token_embedding.T, len(self.dictionary))

        word_similarity_index = WordEmbeddingSimilarityIndex(embeddings)

        self.similarity_matrix = SparseTermSimilarityMatrix(word_similarity_index, self.dictionary)

        def get_matching_tokens(augmented_tokens: Iterable[Tuple[Any, str]],
                                searched_token: str) -> Iterable[Tuple[Any, str]]:
            for augmented_token in augmented_tokens:
                if unaugment_token(augmented_token) == searched_token:
                    yield augmented_token

        def unaugment_token(augmented_token: Tuple[Any, str]) -> str:
            return augmented_token.split()[-1]

        # Convert to a sparse matrix type that allows modification
        matrix = dok_matrix(self.similarity_matrix.matrix)

        for augm_ref_tokens, augm_trans_tokens in tqdm(self.zipped_test_corpus,
                                                       desc=f'{self.label}: patch similarity matrix'):
            shared_tokens = set(map(unaugment_token, augm_ref_tokens + augm_trans_tokens))
            for shared_token in shared_tokens:
                matching_augm_ref_tokens = get_matching_tokens(augm_ref_tokens, shared_token)
                matching_augm_trans_tokens = get_matching_tokens(augm_trans_tokens, shared_token)
                all_pairs = product(matching_augm_ref_tokens, matching_augm_trans_tokens)
                for token_pair in all_pairs:
                    matching_indexes = tuple(self.dictionary.token2id[augm_token] for augm_token in token_pair)
                    matrix[matching_indexes] = 1.0

        # Convert back to a sparse matrix type that allows dot products
        self.similarity_matrix.matrix = csr_matrix(matrix)

    def compute(self, judgements: Judgements) -> List[float]:
        # https://stackoverflow.com/questions/59573454/soft-cosine-similarity-between-two-sentences
        out_scores = []
        for reference_words, translation_words in tqdm(self.zipped_test_corpus, desc=self.label):
            ref_index = self.dictionary.doc2bow(reference_words)
            trans_index = self.dictionary.doc2bow(translation_words)
            out_scores.append(self.similarity_matrix.inner_product(ref_index, trans_index, normalized=(True, True)))

        return out_scores


class SCM(Metric):
    label = "SCM"
    w2v_model = None
    stopwords = None
    similarity_matrix = None
    dictionary = None
    tfidf = None

    def __init__(self, tgt_lang: str, use_tfidf: bool):
        if tgt_lang == "en":
            self.w2v_model = load_facebook_vectors('embeddings/cc.en.300.bin')
            self.w2v_model.init_sims(replace=True)
            nltk.download('stopwords')
            self.stopwords = stopwords.words('english')
        else:
            raise ValueError(tgt_lang)

        self.use_tfidf = use_tfidf
        if use_tfidf:
            self.label = self.label + "_tfidf"

    def fit(self, train_judgements: Judgements, test_judgements: Judgements):
        self.dictionary = Dictionary([[t.lower() for t in simple_preprocess(refs[0])] for refs in test_judgements.references])
        similarity_index = WordEmbeddingSimilarityIndex(self.w2v_model)

        if self.use_tfidf:
            from gensim.models import TfidfModel
            self.tfidf = TfidfModel(dictionary=self.dictionary)
            self.similarity_matrix = SparseTermSimilarityMatrix(similarity_index, self.dictionary, self.tfidf)
        else:
            self.similarity_matrix = SparseTermSimilarityMatrix(similarity_index, self.dictionary)

    def compute(self, judgements: Judgements, threshold_importance: float = 0) -> List[float]:
        # https://stackoverflow.com/questions/59573454/soft-cosine-similarity-between-two-sentences
        out_scores = []
        for reference, translation in tqdm(zip(judgements.references, judgements.translations),
                                           desc="SCM", total=len(judgements)):
            reference_words = [w.lower() for w in simple_preprocess(reference[0]) if w.lower() not in self.stopwords]
            translation_words = [w.lower() for w in simple_preprocess(translation) if w.lower() not in self.stopwords]

            if self.use_tfidf:
                ref_index = self.tfidf[self.dictionary.doc2bow(reference_words)]
                trans_index = self.tfidf[self.dictionary.doc2bow(translation_words)]

                if threshold_importance:
                    # take only the top 50% most-important terms
                    safe_sum = len(ref_index + trans_index) if len(ref_index + trans_index) > 0 else 1
                    threshold_tfidf = sum([val for idx, val in ref_index + trans_index]) / safe_sum
                    ref_index = [(idx, val) for idx, val in ref_index if val >= threshold_tfidf]
                    trans_index = [(idx, val) for idx, val in trans_index if val >= threshold_tfidf]
            else:
                ref_index = self.dictionary.doc2bow(reference_words)
                trans_index = self.dictionary.doc2bow(translation_words)

            out_scores.append(self.similarity_matrix.inner_product(ref_index, trans_index, normalized=(True, True)))

        return out_scores
