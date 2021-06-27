from typing import List
from itertools import product, chain

from gensim.corpora import Dictionary
from gensim.similarities import WordEmbeddingSimilarityIndex, SparseTermSimilarityMatrix
from nltk.corpus import stopwords
from gensim.models.fasttext import load_facebook_vectors
from gensim.models.keyedvectors import KeyedVectors, _add_word_to_kv
import nltk
from tqdm import tqdm
from scipy.sparse import dok_matrix, csr_matrix

from common import Metric, Judgements, AugmentedCorpus
from embedder import ContextualEmbedder


class ContextualSCM(Metric):
    label = "SCM_contextual"
    stopwords = None
    similarity_matrix = None
    dictionary = None
    tfidf = None
    embedder = None
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
        self.dictionary = Dictionary(corpus)

        embeddings = KeyedVectors(self.embedder.vector_size, len(self.dictionary), dtype=float)
        for augmented_tokens, tokens_embeddings in tqdm(zip(corpus, corpus_embeddings),
                                                        desc=f'{self.label}: construct contextual embeddings'):
            for token_index, (token, token_embedding) in enumerate(zip(augmented_tokens, tokens_embeddings)):
                _add_word_to_kv(embeddings, None, token, token_embedding.T, len(self.dictionary))

        word_similarity_index = WordEmbeddingSimilarityIndex(embeddings)

        self.similarity_matrix = SparseTermSimilarityMatrix(word_similarity_index, self.dictionary)

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
        reference_corpus, translation_corpus = map(
            list, zip(*test_judgements.get_tokenized_texts(self.stopwords, desc=self.label)))
        corpus = reference_corpus + translation_corpus

        # We only use words from test corpus, since we don't care about words from train corpus
        self.dictionary = Dictionary(corpus)
        similarity_index = WordEmbeddingSimilarityIndex(self.w2v_model)

        if self.use_tfidf:
            from gensim.models import TfidfModel
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
