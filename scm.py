from collections import defaultdict
from typing import List, Any
from itertools import product, chain
from functools import lru_cache

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

from common import ReferenceFreeMetric, Metric, Judgements, AugmentedCorpus
from embedder import ContextualEmbedder


class ContextualSCM(ReferenceFreeMetric):
    label = "SCM_contextual"
    embedder = None

    def __init__(self, tgt_lang: str, reference_free: bool = False):
        self.embedder = ContextualEmbedder(lang=tgt_lang, reference_free=reference_free)
        self.reference_free = reference_free

    @lru_cache(maxsize=None)
    def compute(self, judgements: Judgements) -> List[float]:
        if self.reference_free:
            ref_corpus, ref_embs = self.embedder.tokenize_embed(list(judgements.src_texts))
        else:
            ref_corpus, ref_embs = self.embedder.tokenize_embed([t[0] for t in judgements.references])
        trans_corpus, trans_embs = self.embedder.tokenize_embed(list(judgements.translations))

        augmented_reference_corpus = AugmentedCorpus('test-reference', ref_corpus)
        augmented_translation_corpus = AugmentedCorpus('test-translation', trans_corpus)

        corpus = augmented_reference_corpus.corpus + augmented_translation_corpus.corpus
        embeddings = ref_embs + trans_embs
        dictionary = Dictionary(corpus, prune_at=None)

        w2v_model = KeyedVectors(self.embedder.vector_size, len(dictionary), dtype=float)
        for augmented_tokens, tokens_embeddings in tqdm(zip(corpus, embeddings),
                                                        desc=f'{self.label}: construct contextual embeddings'):
            for token, token_embedding in zip(augmented_tokens, tokens_embeddings):
                _add_word_to_kv(w2v_model, None, token, token_embedding, len(dictionary))

        annoy = AnnoyIndexer(w2v_model, num_trees=1)
        similarity_index = WordEmbeddingSimilarityIndex(w2v_model, kwargs={'indexer': annoy})
        similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary)

        # Convert to a sparse matrix type that allows modification
        matrix = dok_matrix(similarity_matrix.matrix)

        zipped_corpus = list(zip(augmented_reference_corpus.corpus, augmented_translation_corpus.corpus))
        for augm_ref_tokens, augm_trans_tokens in tqdm(zipped_corpus,
                                                       desc=f'{self.label}: patch similarity matrix'):
            shared_tokens = set(chain(
                map(augmented_reference_corpus.unaugment_token, augm_ref_tokens),
                map(augmented_translation_corpus.unaugment_token, augm_trans_tokens),
            ))
            for shared_token in shared_tokens:
                matching_augm_ref_tokens = augmented_reference_corpus.get_matching_tokens(
                    augm_ref_tokens, shared_token)
                matching_augm_trans_tokens = augmented_reference_corpus.get_matching_tokens(
                    augm_trans_tokens, shared_token)
                all_pairs = product(matching_augm_ref_tokens, matching_augm_trans_tokens)
                for token_pair in all_pairs:
                    matching_indexes = tuple(dictionary.token2id[augm_token] for augm_token in token_pair)
                    matrix[matching_indexes] = 1.0

        # Convert back to a sparse matrix type that allows dot products
        similarity_matrix.matrix = csr_matrix(matrix)

        out_scores = []
        for reference_words, translation_words in tqdm(zipped_corpus, desc=self.label):
            ref_index = dictionary.doc2bow(reference_words)
            trans_index = dictionary.doc2bow(translation_words)
            out_scores.append(similarity_matrix.inner_product(ref_index, trans_index, normalized=(True, True)))

        return out_scores

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ContextualSCM):
            return NotImplemented
        return all([
            self.reference_free == other.reference_free,
            self.embedder == other.embedder,
        ])

    def __hash__(self) -> int:
        return hash((self.reference_free, self.embedder))


class DecontextualizedSCM(ReferenceFreeMetric):

    label = "SCM_decontextualized"

    def __init__(self, tgt_lang: str, use_tfidf: bool, reference_free: bool = False):
        self.embedder = ContextualEmbedder(lang=tgt_lang, reference_free=reference_free)
        self.reference_free = reference_free

        self.use_tfidf = use_tfidf
        if use_tfidf:
            self.label = self.label + "_tfidf"

    @lru_cache(maxsize=None)
    def compute(self, judgements: Judgements) -> List[float]:
        if self.reference_free:
            ref_corpus, ref_embs = self.embedder.tokenize_embed(list(judgements.src_texts))
        else:
            ref_corpus, ref_embs = self.embedder.tokenize_embed([t[0] for t in judgements.references])
        trans_corpus, trans_embs = self.embedder.tokenize_embed(list(judgements.translations))

        corpus = ref_corpus + trans_corpus
        embeddings = ref_embs + trans_embs
        dictionary = Dictionary(corpus)

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
        annoy = AnnoyIndexer(w2v_model, num_trees=1)
        similarity_index = WordEmbeddingSimilarityIndex(w2v_model, kwargs={'indexer': annoy})

        if self.use_tfidf:
            tfidf = TfidfModel(dictionary=dictionary)
            similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary, tfidf)
        else:
            similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary)

        out_scores = []
        zipped_corpus = list(zip(ref_corpus, trans_corpus))
        for reference_words, translation_words in tqdm(zipped_corpus, desc=self.label):
            ref_index = dictionary.doc2bow(reference_words)
            trans_index = dictionary.doc2bow(translation_words)
            if self.use_tfidf:
                ref_index = tfidf[ref_index]
                trans_index = tfidf[trans_index]
            out_scores.append(similarity_matrix.inner_product(ref_index, trans_index, normalized=(True, True)))

        return out_scores

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DecontextualizedSCM):
            return NotImplemented
        return all([
            self.reference_free == other.reference_free,
            self.embedder == other.embedder,
            self.use_tfidf == other.use_tfidf,
        ])

    def __hash__(self) -> int:
        return hash((self.reference_free, self.embedder, self.use_tfidf))


class SCM(Metric):
    label = "SCM"
    w2v_model = None
    stopwords = None

    def __init__(self, tgt_lang: str, use_tfidf: bool):
        if tgt_lang == "en":
            self.w2v_model = load_facebook_vectors('embeddings/cc.en.300.bin')
            self.w2v_model.init_sims(replace=True)
            nltk.download('stopwords', quiet=True)
            self.stopwords = stopwords.words('english')
        else:
            raise ValueError(tgt_lang)

        self.use_tfidf = use_tfidf
        if use_tfidf:
            self.label = self.label + "_tfidf"

    @lru_cache(maxsize=None)
    def compute(self, judgements: Judgements) -> List[float]:
        ref_corpus, trans_corpus = map(
            list, zip(*judgements.get_tokenized_texts(self.stopwords, desc=self.label)))

        corpus = ref_corpus + trans_corpus
        dictionary = Dictionary(corpus)

        annoy = AnnoyIndexer(self.w2v_model, num_trees=1)
        similarity_index = WordEmbeddingSimilarityIndex(self.w2v_model, kwargs={'indexer': annoy})

        if self.use_tfidf:
            tfidf = TfidfModel(dictionary=dictionary)
            similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary, tfidf)
        else:
            similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary)

        out_scores = []
        for reference_words, translation_words in judgements.get_tokenized_texts(self.stopwords, desc=self.label):
            ref_index = dictionary.doc2bow(reference_words)
            trans_index = dictionary.doc2bow(translation_words)
            if self.use_tfidf:
                ref_index = tfidf[ref_index]
                trans_index = tfidf[trans_index]
            out_scores.append(similarity_matrix.inner_product(ref_index, trans_index, normalized=(True, True)))
        return out_scores

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SCM):
            return NotImplemented
        return all([
            self.use_tfidf == other.use_tfidf,
        ])

    def __hash__(self) -> int:
        return hash(self.use_tfidf)
