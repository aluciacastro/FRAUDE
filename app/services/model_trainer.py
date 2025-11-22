"""
Servicio para entrenamiento de modelos
Principio SOLID: Single Responsibility - Solo entrena y evalúa modelos
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from app.models.ml_models import ModelFactory, ModelEvaluator
from app.services.data_preprocessor import BalancingStrategyFactory
import joblib
from pathlib import Path
from config.settings import MODELS_DIR


class ModelTrainer:
    """Clase responsable del entrenamiento de modelos"""

    def __init__(self):
        self.trained_models = {}
        self.results = []
        self.evaluator = ModelEvaluator()

    def train_single_model(
        self,
        model_name: str,
        X_train,
        y_train,
        X_test,
        y_test,
        strategy_name: str = 'Original'
    ) -> dict:
        """
        Entrenar un modelo específico

        Args:
            model_name: Nombre del modelo a entrenar
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            X_test: Features de prueba
            y_test: Target de prueba
            strategy_name: Estrategia de balanceo usada

        Returns:
            Diccionario con resultados de la evaluación
        """
        print(f"\nEntrenando {model_name} con estrategia {strategy_name}...")

        # Crear y entrenar modelo
        model = ModelFactory.create_model_by_name(model_name)
        model.train(X_train, y_train)

        # Evaluar modelo
        results = self.evaluator.evaluate_model(model, X_test, y_test)
        results['strategy'] = strategy_name

        # Almacenar modelo entrenado
        key = f"{model_name}_{strategy_name}"
        self.trained_models[key] = model

        print(f"  Accuracy: {results['metrics']['accuracy']:.4f}")
        print(f"  Precision: {results['metrics']['precision']:.4f}")
        print(f"  Recall: {results['metrics']['recall']:.4f}")
        print(f"  F1-Score: {results['metrics']['f1_score']:.4f}")
        print(f"  ROC-AUC: {results['metrics']['roc_auc']:.4f}")

        return results

    def train_all_models(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        strategy_name: str = 'Original',
        model_names: List[str] = None
    ) -> List[dict]:
        """
        Entrenar múltiples modelos

        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            X_test: Features de prueba
            y_test: Target de prueba
            strategy_name: Estrategia de balanceo usada
            model_names: Lista de nombres de modelos (None para todos)

        Returns:
            Lista de resultados de evaluación
        """
        if model_names is None:
            model_names = [
                'Logistic Regression',
                'Decision Tree',
                'Random Forest',
                'Gradient Boosting'
            ]

        print("="*60)
        print(f"ENTRENAMIENTO CON ESTRATEGIA: {strategy_name}")
        print("="*60)

        strategy_results = []

        for model_name in model_names:
            result = self.train_single_model(
                model_name, X_train, y_train, X_test, y_test, strategy_name
            )
            strategy_results.append(result)

        self.results.extend(strategy_results)
        return strategy_results

    def train_with_multiple_strategies(
        self,
        X_train_original,
        y_train_original,
        X_test,
        y_test,
        strategies: List[str] = None
    ) -> Dict[str, List[dict]]:
        """
        Entrenar modelos con múltiples estrategias de balanceo

        Args:
            X_train_original: Features de entrenamiento originales
            y_train_original: Target de entrenamiento original
            X_test: Features de prueba
            y_test: Target de prueba
            strategies: Lista de estrategias (None para todas)

        Returns:
            Diccionario con resultados por estrategia
        """
        if strategies is None:
            strategies = ['Original', 'SMOTE', 'Under-sampling']

        all_results = {}

        for strategy_name in strategies:
            # Aplicar estrategia de balanceo
            if strategy_name == 'Original':
                X_train = X_train_original
                y_train = y_train_original
            else:
                strategy = BalancingStrategyFactory.create_strategy(strategy_name)
                X_train, y_train = strategy.balance(X_train_original, y_train_original)

            # Entrenar modelos con esta estrategia
            strategy_results = self.train_all_models(
                X_train, y_train, X_test, y_test, strategy_name
            )
            all_results[strategy_name] = strategy_results

        return all_results

    def get_best_model(self, metric: str = 'f1_score') -> Tuple[dict, str]:
        """
        Obtener el mejor modelo según una métrica

        Args:
            metric: Métrica a usar para comparación

        Returns:
            Tupla con (resultado del mejor modelo, clave del modelo)
        """
        if not self.results:
            raise ValueError("No hay modelos entrenados")

        best_result = max(self.results, key=lambda x: x['metrics'][metric])
        best_key = f"{best_result['model_name']}_{best_result['strategy']}"

        return best_result, best_key

    def compare_all_models(self) -> pd.DataFrame:
        """
        Comparar todos los modelos entrenados

        Returns:
            DataFrame con comparación de modelos
        """
        return self.evaluator.compare_models(self.results)

    def save_model(self, model_key: str, file_path: Path = None):
        """
        Guardar un modelo entrenado

        Args:
            model_key: Clave del modelo en trained_models
            file_path: Ruta donde guardar (None para ruta por defecto)
        """
        if model_key not in self.trained_models:
            raise ValueError(f"Modelo no encontrado: {model_key}")

        if file_path is None:
            file_path = MODELS_DIR / f"{model_key}.joblib"

        joblib.dump(self.trained_models[model_key], file_path)
        print(f" Modelo guardado: {file_path}")

    def save_best_model(self, metric: str = 'f1_score') -> Path:
        """
        Guardar el mejor modelo

        Args:
            metric: Métrica a usar para seleccionar el mejor

        Returns:
            Ruta donde se guardó el modelo
        """
        best_result, best_key = self.get_best_model(metric)
        file_path = MODELS_DIR / f"best_model_{metric}.joblib"
        self.save_model(best_key, file_path)

        print(f"\nMejor modelo ({metric}): {best_result['model_name']}")
        print(f"Estrategia: {best_result['strategy']}")
        print(f"{metric}: {best_result['metrics'][metric]:.4f}")

        return file_path

    @staticmethod
    def load_model(file_path: Path):
        """
        Cargar un modelo guardado

        Args:
            file_path: Ruta del modelo

        Returns:
            Modelo cargado
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Modelo no encontrado: {file_path}")

        model = joblib.load(file_path)
        print(f" Modelo cargado: {file_path}")
        return model


class ExperimentTracker:
    """Clase para rastrear experimentos de entrenamiento"""

    def __init__(self):
        self.experiments = []

    def add_experiment(
        self,
        experiment_name: str,
        results: List[dict],
        metadata: dict = None
    ):
        """
        Agregar un experimento

        Args:
            experiment_name: Nombre del experimento
            results: Resultados del experimento
            metadata: Información adicional
        """
        experiment = {
            'name': experiment_name,
            'results': results,
            'metadata': metadata or {},
            'timestamp': pd.Timestamp.now()
        }
        self.experiments.append(experiment)

    def get_experiment_summary(self) -> pd.DataFrame:
        """
        Obtener resumen de todos los experimentos

        Returns:
            DataFrame con resumen
        """
        summary_data = []

        for exp in self.experiments:
            for result in exp['results']:
                summary_data.append({
                    'Experiment': exp['name'],
                    'Model': result['model_name'],
                    'Strategy': result['strategy'],
                    'F1-Score': result['metrics']['f1_score'],
                    'ROC-AUC': result['metrics']['roc_auc'],
                    'Recall': result['metrics']['recall'],
                    'Precision': result['metrics']['precision'],
                    'Timestamp': exp['timestamp']
                })

        return pd.DataFrame(summary_data)

    def get_best_across_experiments(self, metric: str = 'f1_score') -> dict:
        """
        Obtener el mejor resultado a través de todos los experimentos

        Args:
            metric: Métrica a usar

        Returns:
            Diccionario con el mejor resultado
        """
        all_results = []
        for exp in self.experiments:
            all_results.extend(exp['results'])

        if not all_results:
            raise ValueError("No hay experimentos registrados")

        best_result = max(all_results, key=lambda x: x['metrics'][metric])
        return best_result
