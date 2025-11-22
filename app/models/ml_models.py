"""
Modelos de Machine Learning
Principio SOLID: Open/Closed - Fácil de extender con nuevos modelos
Principio SOLID: Liskov Substitution - Todos los modelos son intercambiables
"""
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, precision_recall_curve, roc_curve,
    f1_score, precision_score, recall_score, accuracy_score
)
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any


class MLModelInterface(ABC):
    """Interface para modelos de ML (Principio: Interface Segregation)"""
    
    @abstractmethod
    def train(self, X_train, y_train):
        """Entrenar el modelo"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Predecir"""
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        """Predecir probabilidades"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Obtener nombre del modelo"""
        pass


class BaseMLModel(MLModelInterface):
    """Clase base para modelos de ML"""
    
    def __init__(self, model, model_name: str):
        """
        Args:
            model: Instancia del modelo de sklearn/xgboost
            model_name: Nombre descriptivo del modelo
        """
        self.model = model
        self.model_name = model_name
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """Entrenar el modelo"""
        print(f"  Entrenando {self.model_name}...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"  ✓ {self.model_name} entrenado")
    
    def predict(self, X):
        """Predecir clases"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predecir probabilidades"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        return self.model.predict_proba(X)
    
    def get_model_name(self) -> str:
        """Obtener nombre del modelo"""
        return self.model_name
    
    def get_feature_importance(self) -> np.ndarray:
        """Obtener importancia de características (si está disponible)"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None


class LogisticRegressionModel(BaseMLModel):
    """Modelo de Regresión Logística"""
    
    def __init__(self, **kwargs):
        model = LogisticRegression(random_state=42, max_iter=1000, **kwargs)
        super().__init__(model, "Logistic Regression")


class RandomForestModel(BaseMLModel):
    """Modelo de Random Forest"""
    
    def __init__(self, **kwargs):
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            **kwargs
        )
        super().__init__(model, "Random Forest")


class XGBoostModel(BaseMLModel):
    """Modelo de XGBoost"""
    
    def __init__(self, **kwargs):
        model = XGBClassifier(
            n_estimators=100,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            **kwargs
        )
        super().__init__(model, "XGBoost")


class GradientBoostingModel(BaseMLModel):
    """Modelo de Gradient Boosting"""
    
    def __init__(self, **kwargs):
        model = GradientBoostingClassifier(
            n_estimators=100,
            random_state=42,
            **kwargs
        )
        super().__init__(model, "Gradient Boosting")


class DecisionTreeModel(BaseMLModel):
    """Modelo de Árbol de Decisión"""
    
    def __init__(self, **kwargs):
        model = DecisionTreeClassifier(random_state=42, **kwargs)
        super().__init__(model, "Decision Tree")


class ModelEvaluator:
    """
    Clase para evaluar modelos
    Principio SOLID: Single Responsibility - Solo evalúa modelos
    """
    
    @staticmethod
    def evaluate_model(model: MLModelInterface, X_test, y_test) -> Dict[str, Any]:
        """
        Evaluar un modelo con múltiples métricas
        
        Args:
            model: Modelo entrenado
            X_test: Features de test
            y_test: Target de test
            
        Returns:
            Diccionario con métricas y datos de evaluación
        """
        # Predicciones
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Métricas básicas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        
        # Curvas ROC y Precision-Recall
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
        
        # Reporte de clasificación
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        results = {
            'model_name': model.get_model_name(),
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc
            },
            'confusion_matrix': cm,
            'roc_curve': {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': roc_thresholds
            },
            'precision_recall_curve': {
                'precision': precision_curve,
                'recall': recall_curve,
                'thresholds': pr_thresholds
            },
            'classification_report': class_report,
            'predictions': y_pred,
            'predictions_proba': y_pred_proba
        }
        
        return results
    
    @staticmethod
    def compare_models(results_list: list) -> pd.DataFrame:
        """
        Comparar múltiples modelos
        
        Args:
            results_list: Lista de resultados de evaluación
            
        Returns:
            DataFrame con comparación de modelos
        """
        comparison_data = []
        
        for result in results_list:
            comparison_data.append({
                'Model': result['model_name'],
                'Accuracy': result['metrics']['accuracy'],
                'Precision': result['metrics']['precision'],
                'Recall': result['metrics']['recall'],
                'F1-Score': result['metrics']['f1_score'],
                'ROC-AUC': result['metrics']['roc_auc']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('F1-Score', ascending=False)
        
        return df_comparison