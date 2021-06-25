from typing import List, Any, Iterable, Tuple
from itertools import product

from gensim.corpora import Dictionary
from gensim.similarities import WordEmbeddingSimilarityIndex, SparseTermSimilarityMatrix
from nltk.corpus import stopwords
from gensim.models.fasttext import load_facebook_vectors
from gensim.models.keyedvectors import KeyedVectors, _add_word_to_kv
import nltk
from tqdm import tqdm
from scipy.sparse import dok_matrix, csr_matrix

from common import Metric, Judgements


class SCM(Metric):
    label = "SCM"
    w2v_model = None
    stopwords = None
    similarity_matrix = None
    dictionary = None
    tfidf = None

    def __init__(self, tgt_lang: str, use_tfidf: bool, use_contextual: bool):
        if tgt_lang == "en":
            nltk.download('stopwords')
            self.stopwords = stopwords.words('english')
        else:
            raise ValueError(tgt_lang)

        self.use_tfidf = use_tfidf
        self.use_contextual = use_contextual

        if use_tfidf:
            self.label = self.label + "_tfidf"
        if use_contextual:
            self.label = self.label + "_contextual"

    def fit(self, train_judgements: Judgements, test_judgements: Judgements):
        test_reference_corpus, test_translation_corpus = map(list, zip(*test_judgements.get_tokenized_texts(self.stopwords)))

        if self.use_contextual:

            def augment_corpus(prefix: Any, corpus: Iterable[List[str]]) -> List[List[Tuple[Any, str]]]:
                return [augment_tokens((prefix, tokens_index), tokens) for tokens_index, tokens in enumerate(corpus)]

            def augment_tokens(prefix: Any, tokens: Iterable[str]) -> List[Tuple[Any, str]]:
                return [((prefix, token_index), token) for token_index, token in enumerate(tokens)]

            augmented_test_reference_corpus = augment_corpus('test-reference', test_reference_corpus)
            augmented_test_translation_corpus = augment_corpus('test-translation', test_translation_corpus)

            self.zipped_test_corpus = list(zip(augmented_test_reference_corpus, augmented_test_translation_corpus))
            corpus = augmented_test_reference_corpus + augmented_test_translation_corpus
        else:
            self.zipped_test_corpus = list(zip(test_reference_corpus, test_translation_corpus))
            corpus = test_reference_corpus + test_translation_corpus

        self.dictionary = Dictionary(corpus)  # We only use only words from test corpus, since we don't care about words from train corpus

        if self.use_contextual:
            embeddings = KeyedVectors(CONTEXTUAL_VECTOR_SIZE, len(self.dictionary), dtype=CONTEXTUAL_VECTOR_DTYPE)  # FIXME: Define the constants
            for augmented_tokens in tqdm(corpus, desc=f'{self.label}: construct contextual embeddings'):
                for token_index, token in enumerate(augmented_tokens):
                    weights = get_contextual_embedding(augmented_tokens, token_index)  # FIXME: Define get_contextual_embedding()
                    _add_word_to_kv(embeddings, None, token, weights, len(self.dictionary))
        else:
            embeddings = load_facebook_vectors('embeddings/cc.en.300.bin').wv

        word_similarity_index = WordEmbeddingSimilarityIndex(embeddings)

        if self.use_tfidf:
            from gensim.models import TfidfModel
            self.tfidf = TfidfModel(dictionary=self.dictionary)
            self.similarity_matrix = SparseTermSimilarityMatrix(word_similarity_index, self.dictionary, self.tfidf)
        else:
            self.similarity_matrix = SparseTermSimilarityMatrix(word_similarity_index, self.dictionary)

        if self.use_contextual:

            def get_matching_tokens(augmented_tokens: Iterable[Tuple[Any, str]],
                                    searched_token: str) -> Iterable[Tuple[Any, str]]:
                for augmented_token in augmented_tokens:
                    if unaugment_token(augmented_token) == searched_token:
                        yield augmented_token

            def unaugment_token(augmented_token: Tuple[Any, str]) -> str:
                prefix, token = augmented_token
                return token

            matrix = dok_matrix(self.similarity_matrix.matrix)  # Convert to a sparse matrix type that allows modification

            for augmented_reference_tokens, augmented_translation_tokens in tqdm(self.zipped_test_corpus,
                                                                                 desc=f'{self.label}: patch similarity matrix'):
                shared_tokens = set(map(unaugment_token, augmented_reference_tokens + augmented_translation_tokens))
                for shared_token in shared_tokens:
                    matching_augmented_reference_tokens = get_matching_tokens(augmented_reference_tokens, shared_token)
                    matching_augmented_translation_tokens = get_matching_tokens(augmented_translation_tokens, shared_token)
                    all_pairs = product(matching_augmented_reference_tokens, matching_augmented_translation_tokens)
                    for matching_augmented_token_pair in all_pairs:
                        matching_indexes = tuple(self.dictionary[augmented_token] for augmented_token in matching_augmented_token_pair)
                        matrix[matching_indexes] = 1.0

            self.similarity_matrix.matrix = csr_matrix(matrix)  # Convert back to a sparse matrix type that allows dot products

    def compute(self, judgements: Judgements) -> List[float]:
        # https://stackoverflow.com/questions/59573454/soft-cosine-similarity-between-two-sentences
        out_scores = []
        for reference_words, translation_words in tqdm(self.zipped_test_corpus, desc=self.label):
            if self.use_tfidf:
                ref_index = self.tfidf[self.dictionary.doc2bow(reference_words)]
                trans_index = self.tfidf[self.dictionary.doc2bow(translation_words)]

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
