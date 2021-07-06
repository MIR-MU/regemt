from collections import defaultdict
from typing import List
from itertools import product, chain

from gensim.corpora import Dictionary
from gensim.similarities import WordEmbeddingSimilarityIndex, SparseTermSimilarityMatrix
from gensim.similarities.annoy import AnnoyIndexer
from nltk.corpus import stopwords
from gensim.models import TfidfModel
from gensim.models.fasttext import load_facebook_vectors
from gensim.models.keyedvectors import KeyedVectors, _add_word_to_kv
import nltk
import numpy as np
from tqdm.autonotebook import tqdm
from scipy.sparse import dok_matrix, csr_matrix

from common import Metric, Judgements, AugmentedCorpus
from embedder import ContextualEmbedder


class ContextualSCM(Metric):
    label = "SCM_contextual"
    similarity_matrix = None
    dictionary = None
    tfidf = None
    embedder = None
    test_judgements = None
    zipped_test_corpus = None

    def __init__(self, tgt_lang: str):
        self.embedder = ContextualEmbedder(lang=tgt_lang)

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
        self.dictionary = Dictionary(corpus, prune_at=None)

        w2v_model = KeyedVectors(self.embedder.vector_size, len(self.dictionary), dtype=float)
        for augmented_tokens, tokens_embeddings in tqdm(zip(corpus, embeddings),
                                                        desc=f'{self.label}: construct contextual embeddings'):
            for token, token_embedding in zip(augmented_tokens, tokens_embeddings):
                _add_word_to_kv(w2v_model, None, token, token_embedding, len(self.dictionary))

        annoy = AnnoyIndexer(w2v_model, num_trees=1)
        similarity_index = WordEmbeddingSimilarityIndex(w2v_model, kwargs={'indexer': annoy})
        self.similarity_matrix = SparseTermSimilarityMatrix(similarity_index, self.dictionary)

        # Convert to a sparse matrix type that allows modification
        matrix = dok_matrix(self.similarity_matrix.matrix)

        for augm_ref_tokens, augm_trans_tokens in tqdm(self.zipped_test_corpus,
                                                       desc=f'{self.label}: patch similarity matrix'):
            shared_tokens = set(chain(
                map(augmented_test_reference_corpus.unaugment_token, augm_ref_tokens),
                map(augmented_test_translation_corpus.unaugment_token, augm_trans_tokens),
            ))
            for shared_token in shared_tokens:
                matching_augm_ref_tokens = augmented_test_reference_corpus.get_matching_tokens(
                    augm_ref_tokens, shared_token)
                matching_augm_trans_tokens = augmented_test_reference_corpus.get_matching_tokens(
                    augm_trans_tokens, shared_token)
                all_pairs = product(matching_augm_ref_tokens, matching_augm_trans_tokens)
                for token_pair in all_pairs:
                    matching_indexes = tuple(self.dictionary.token2id[augm_token] for augm_token in token_pair)
                    matrix[matching_indexes] = 1.0

        # Convert back to a sparse matrix type that allows dot products
        self.similarity_matrix.matrix = csr_matrix(matrix)

    def compute(self, judgements: Judgements) -> List[float]:
        if judgements != self.test_judgements:
            raise ValueError('Tne judgements are different from the test_judgements used in fit()')

        # https://stackoverflow.com/questions/59573454/soft-cosine-similarity-between-two-sentences
        out_scores = []
        for reference_words, translation_words in tqdm(self.zipped_test_corpus, desc=self.label):
            ref_index = self.dictionary.doc2bow(reference_words)
            trans_index = self.dictionary.doc2bow(translation_words)
            out_scores.append(self.similarity_matrix.inner_product(ref_index, trans_index, normalized=(True, True)))

        return out_scores


class DecontextualizedSCM(Metric):

    label = "SCM_decontextualized"
    test_judgements = None
    zipped_test_corpus = None
    similarity_matrix = None
    dictionary = None
    tfidf = None

    def __init__(self, tgt_lang: str, use_tfidf: bool):
        self.embedder = ContextualEmbedder(lang=tgt_lang)
        if tgt_lang != "en":
            raise ValueError(tgt_lang)

        self.use_tfidf = use_tfidf
        if use_tfidf:
            self.label = self.label + "_tfidf"

    def fit(self, train_judgements: Judgements, test_judgements: Judgements):
        self.test_judgements = test_judgements

        test_ref_corpus, test_ref_embs = self.embedder.tokenize_embed([t[0] for t in test_judgements.references])
        test_trans_corpus, test_trans_embs = self.embedder.tokenize_embed(test_judgements.translations)

        self.zipped_test_corpus = list(zip(test_ref_corpus, test_trans_corpus))

        # We only use words from test corpus, since we don't care about words from train corpus
        corpus = test_ref_corpus + test_trans_corpus
        embeddings = test_ref_embs + test_trans_embs
        self.dictionary = Dictionary(corpus)

        # We average embeddings for all occurences for a term
        decontextualized_embeddings = defaultdict(lambda: [])
        for tokens, tokens_embeddings in zip(corpus, embeddings):
            for token, token_embedding in zip(tokens, tokens_embeddings):
                decontextualized_embeddings[token].append(token_embedding)

        w2v_model = KeyedVectors(self.embedder.vector_size, len(decontextualized_embeddings), dtype=float)
        for token, token_embeddings in tqdm(decontextualized_embeddings.items(),
                                            f'{self.label}: construct decontextualized embeddings'):
            token_embedding = np.mean(token_embeddings, axis=0)
            _add_word_to_kv(w2v_model, None, token, token_embedding, len(decontextualized_embeddings))
        annoy = AnnoyIndexer(w2v_model, num_trees=1)
        similarity_index = WordEmbeddingSimilarityIndex(w2v_model, kwargs={'indexer': annoy})

        if self.use_tfidf:
            self.tfidf = TfidfModel(dictionary=self.dictionary)
            self.similarity_matrix = SparseTermSimilarityMatrix(similarity_index, self.dictionary, self.tfidf)
        else:
            self.similarity_matrix = SparseTermSimilarityMatrix(similarity_index, self.dictionary)

    def compute(self, judgements: Judgements) -> List[float]:
        if judgements != self.test_judgements:
            raise ValueError('Tne judgements are different from the test_judgements used in fit()')

        # https://stackoverflow.com/questions/59573454/soft-cosine-similarity-between-two-sentences
        out_scores = []
        for reference_words, translation_words in tqdm(self.zipped_test_corpus, desc=self.label):
            ref_index = self.dictionary.doc2bow(reference_words)
            trans_index = self.dictionary.doc2bow(translation_words)
            if self.use_tfidf:
                ref_index = self.tfidf[ref_index]
                trans_index = self.tfidf[trans_index]
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
        test_ref_corpus, test_trans_corpus = map(
            list, zip(*test_judgements.get_tokenized_texts(self.stopwords, desc=self.label)))

        # We only use words from test corpus, since we don't care about words from train corpus
        corpus = test_ref_corpus + test_trans_corpus
        self.dictionary = Dictionary(corpus)

        annoy = AnnoyIndexer(self.w2v_model, num_trees=1)
        similarity_index = WordEmbeddingSimilarityIndex(self.w2v_model, kwargs={'indexer': annoy})

        if self.use_tfidf:
            self.tfidf = TfidfModel(dictionary=self.dictionary)
            self.similarity_matrix = SparseTermSimilarityMatrix(similarity_index, self.dictionary, self.tfidf)
        else:
            self.similarity_matrix = SparseTermSimilarityMatrix(similarity_index, self.dictionary)

    def compute(self, judgements: Judgements) -> List[float]:
        # https://stackoverflow.com/questions/59573454/soft-cosine-similarity-between-two-sentences
        out_scores = []
        for reference_words, translation_words in judgements.get_tokenized_texts(self.stopwords, desc=self.label):
            ref_index = self.dictionary.doc2bow(reference_words)
            trans_index = self.dictionary.doc2bow(translation_words)
            if self.use_tfidf:
                ref_index = self.tfidf[ref_index]
                trans_index = self.tfidf[trans_index]
            out_scores.append(self.similarity_matrix.inner_product(ref_index, trans_index, normalized=(True, True)))
        return out_scores
