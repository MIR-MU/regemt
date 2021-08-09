import logging
from typing import Iterable, List, Tuple, Optional, Any
from functools import lru_cache
import warnings

from common import Metric, ReferenceFreeMetric, Judgements
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import RegressorMixin as Model
from sklearn.feature_selection import RFECV
from sklearn.linear_model import (
    LinearRegression,
    SGDRegressor,
    Ridge,
    BayesianRidge,
)
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.utils import parallel_backend
from sklearn.svm import LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from tqdm.autonotebook import tqdm


LOGGER = logging.getLogger(__name__)

Feature = float
Features = Tuple[Feature, ...]

Scores = List[float]


class Regression(ReferenceFreeMetric):

    label = "Regression"
    model: Optional[Model] = None
    judgements: Optional[Judgements] = None
    imputer: Optional[IterativeImputer] = None

    def __init__(self, metrics: Optional[Iterable[Metric]], reference_free: bool = False):
        if metrics is None:
            self.label = self.label + '_baseline'
            self.metrics = None
        else:
            if reference_free:
                metrics = [metric for metric in metrics if isinstance(metric, ReferenceFreeMetric)]
            self.metrics = tuple(metrics)
        self.reference_free = reference_free

    def _get_metric_features(self, judgements: Judgements) -> List[Features]:
        if self.metrics is None:
            return len(judgements) * [()]
        metric_features_transposed = []
        for metric in self.metrics:
            if self.reference_free:
                assert isinstance(metric, ReferenceFreeMetric)
                results = metric.compute_ref_free(judgements)
            else:
                results = metric.compute(judgements)
            if len(results) != len(judgements):
                message = f'{metric}{".compute_ref_free()" if self.reference_free else ".compute()"}'
                message += f' returned {len(results)} results, {len(judgements)} expected'
                raise ValueError(message)
            metric_features_transposed.append(results)
        metric_features = list(zip(*metric_features_transposed))
        return metric_features

    def _get_other_features(self, judgements: Judgements) -> List[Features]:
        other_features = []
        sources = judgements.src_texts if self.reference_free else [t[0] for t in judgements.references]
        for source, translation in zip(sources, judgements.translations):
            other_features.append((float(len(source)), float(len(translation))))
        return other_features

    def _get_features(self, judgements: Judgements, random_state: int = 42,
                      fit_imputer: bool = False) -> List[Features]:
        metric_features_list = self._get_metric_features(judgements)
        other_features_list = self._get_other_features(judgements)
        features = []
        for metric_features, other_features in zip(metric_features_list, other_features_list):
            features.append((*metric_features, *other_features))

        features_array = np.array(features, dtype=float)
        finite_indices = np.isfinite(features_array)
        num_non_finite = len(features) - np.sum(np.all(finite_indices, axis=1))
        features_array[~finite_indices] = np.nan

        if fit_imputer:
            print(f'{self}: fitting an imputer on {len(features)} samples of {features_array.shape[1]} features')
            self.imputer = IterativeImputer(random_state=random_state).fit(features_array)

        if num_non_finite:
            if self.imputer is not None:
                imputed_features = list(map(tuple, self.imputer.transform(features_array)))
                assert len(imputed_features) == len(features)
                LOGGER.warning(f'Imputed {num_non_finite} out of {len(features)} samples with non-finite values')
                features = imputed_features
            else:
                msg = f'{num_non_finite} out of {len(features)} samples contain non-finite values, but no fit imputer'
                raise ValueError(msg)

        return features

    def _get_scores(self, judgements: Judgements) -> Scores:
        return list(judgements.scores)

    def _get_features_and_scores(self, judgements, random_state: int = 42,
                                 fit_imputer: bool = False) -> Tuple[List[Features], Scores]:
        X, y = self._get_features(judgements), self._get_scores(judgements)
        assert len(X) == len(y)

        num_non_finite = np.sum(np.all(np.isfinite(X), axis=0))
        array_X = np.array(X, dtype=np.float64)
        array_X[~np.isfinite(array_X)] = np.nan

        if fit_imputer:
            self.imputer = IterativeImputer(random_state=random_state).fit(array_X)

        if num_non_finite:
            if self.imputer is not None:
                imputed_X = list(map(tuple, self.imputer.transform(array_X)))
                assert len(X) == len(imputed_X)
                LOGGER.warning(f'Imputed {num_non_finite} samples out of {len(X)} with non-finite values')
                X = imputed_X
            else:
                message = f'{num_non_finite} samples out of {len(X)} contain non-finite values, but no imputer exists'
                raise ValueError(message)

        return (X, y)

    def _get_models(self, select_features: bool = False,
                    optimize_hyperparameters: bool = False,
                    random_state: float = 42) -> Iterable[Model]:

        def linear_regression():
            return {
                'model': LinearRegression(),
                'hyperparameters': {
                    'normalize': [True, False],
                    'positive': [True, False],
                },
                'can_select_features': True,
            }

        def sgd_regressor():
            return {
                'model': SGDRegressor(random_state=random_state),
                'hyperparameters': {
                    'penalty': ['l2', 'l1', 'elasticnet'],
                    'early_stopping': [True, False],
                },
                'can_select_features': True,
            }

        def ridge():
            return {
                'model': Ridge(random_state=random_state),
                'hyperparameters': {
                    'alpha': np.logspace(1, 4, 10),
                },
                'can_select_features': True,
            }

        def bayesian_ridge():
            return {
                'model': BayesianRidge(),
                'hyperparameters': None,
                'can_select_features': True,
            }

        def svr():
            return {
                'model': LinearSVR(dual=False, loss='squared_epsilon_insensitive'),
                'hyperparameters': {
                    'C': np.logspace(-2, 3, 10),
                },
                'can_select_features': True,
            }

        def k_nearest_neighbors_regressor():
            return {
                'model': KNeighborsRegressor(),
                'hyperparameters': {
                    'n_neighbors': range(1, 20, 2),
                },
                'can_select_features': False,
            }

        def mlp_regressor():
            return {
                'model': MLPRegressor(random_state=random_state, max_iter=15000),
                'hyperparameters': {
                    'solver': ['lbfgs', 'sgd', 'adam'],
                    'alpha': np.logspace(1, 4, 10),
                },
                'can_select_features': False,
            }

        models = [
            linear_regression(),
            sgd_regressor(),
            ridge(),
            bayesian_ridge(),
            svr(),
            k_nearest_neighbors_regressor(),
            mlp_regressor(),
        ]

        for model in tqdm(models, desc=f'{self}: model selection'):
            estimator = model['model']
            if optimize_hyperparameters and model['hyperparameters'] is not None:
                estimator = GridSearchCV(estimator, model['hyperparameters'])
                importance_getter = 'best_estimator_.coef_'
            else:
                importance_getter = 'auto'
            if select_features and model['can_select_features']:
                estimator = RFECV(estimator, importance_getter=importance_getter)
            yield make_pipeline(StandardScaler(), estimator)

    def fit(self, judgements: Judgements):
        print(f'{self}: getting features on train judgements')
        X, y = self._get_features(judgements, fit_imputer=True), self._get_scores(judgements)

        (train_judgements, [train_X, train_y]), (test_judgements, [test_X, test_y]) = judgements.split(X, y)
        assert (len(train_X), len(train_y)) == (len(train_judgements), len(train_judgements))
        assert (len(test_X), len(test_y)) == (len(test_judgements), len(test_judgements))

        models, best_model, best_r2 = self._get_models(), None, float('-inf')
        with parallel_backend('multiprocessing', n_jobs=32), warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            for model in models:
                model.fit(train_X, train_y)
                r2 = model.score(test_X, test_y)
                if r2 > best_r2:
                    best_model, best_r2 = model, r2
            assert best_model is not None

            print(f'{self}: fitting the selected model')
            best_model.fit(X, y)

        self.judgements = judgements
        self.model = best_model

    @lru_cache(maxsize=None)
    def compute(self, judgements: Judgements) -> List[float]:
        if self.model is None:
            raise ValueError('Using compute() before fit()')
        if self.judgements.overlaps(judgements):
            raise ValueError('Train and test judgements overlap')

        print(f'{self}: getting features on test judgements')
        X = self._get_features(judgements)

        print(f'{self}: making predictions with the selected model')
        y = self.model.predict(X)
        return y

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Regression):
            return NotImplemented
        return all([
            self.reference_free == other.reference_free,
            self.metrics == other.metrics,
            self.judgements == other.judgements,
        ])

    def __hash__(self) -> int:
        return hash((self.reference_free, self.metrics, self.judgements))
