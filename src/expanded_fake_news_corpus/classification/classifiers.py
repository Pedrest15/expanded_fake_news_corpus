"""Catálogo de classificadores e suas grades de busca.

Mesmos estimadores e mesmas grades do trabalho anterior
(``linguistic_features/linguistic_features.py``), para que os resultados das
duas bases sejam comparáveis. O XGBoost entra só se estiver instalado — lá a
dependência também é opcional.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

#: Semente de tudo que amostra, para a execução ser reproduzível.
RANDOM_STATE = 42


@dataclass(frozen=True)
class ClassifierSpec:
    """Um estimador com a grade de hiperparâmetros a varrer."""

    name: str
    estimator: Any
    param_grid: dict[str, list[Any]] = field(default_factory=dict)

    @property
    def grid_size(self) -> int:
        """Número de combinações da grade."""
        total = 1
        for values in self.param_grid.values():
            total *= len(values)
        return total


def build_catalogue() -> dict[str, ClassifierSpec]:
    """Monta o catálogo de classificadores disponíveis.

    Returns:
        Especificação por nome, na ordem em que os relatórios as listam
    """
    catalogue = {
        "svm": ClassifierSpec(
            "svm",
            SVC(probability=True, random_state=RANDOM_STATE),
            {
                "kernel": ["rbf", "linear", "poly"],
                "C": [0.1, 1.0, 10.0, 100.0],
                "gamma": ["scale", "auto"],
            },
        ),
        "random_forest": ClassifierSpec(
            "random_forest",
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        ),
        "naive_bayes": ClassifierSpec(
            "naive_bayes",
            GaussianNB(),
            {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},
        ),
        "logistic_regression": ClassifierSpec(
            "logistic_regression",
            LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            {
                "C": [0.01, 0.1, 1.0, 10.0, 100.0],
                # O trabalho anterior varre penalty=['l1','l2']. O scikit-learn
                # 1.8 depreciou `penalty` em favor de `l1_ratio`, e a tradução
                # é exata — l1_ratio=1 é L1 puro, 0 é L2 puro —, então o espaço
                # de busca continua o mesmo e some o aviso de remoção.
                "l1_ratio": [1.0, 0.0],
                "solver": ["liblinear", "saga"],
            },
        ),
        "mlp": ClassifierSpec(
            "mlp",
            MLPClassifier(random_state=RANDOM_STATE, max_iter=500, early_stopping=True),
            {
                "hidden_layer_sizes": [(128,), (256,), (128, 64), (256, 128)],
                "activation": ["relu", "tanh"],
                "alpha": [0.0001, 0.001, 0.01],
            },
        ),
    }

    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.info("XGBoost não instalado; fica fora do catálogo")
    else:
        catalogue["xgboost"] = ClassifierSpec(
            "xgboost",
            XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1),
            {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 10],
                "learning_rate": [0.01, 0.1, 0.3],
                "subsample": [0.8, 1.0],
            },
        )

    return catalogue
