"""
Pipeline de Detección de Fraude
Principio SOLID: Single Responsibility - Orquesta el flujo completo
Principio SOLID: Dependency Inversion - Depende de abstracciones, no de implementaciones
"""
from typing import Dict, List, Any, Tuple
import pandas as pd
import joblib
from pathlib import Path

from app.services.data_loader import DataLoader as CSVDataLoader, DataLoader
from app.services.data_preprocessor import DataPreprocessor
from app.models.ml_models import (
    LogisticRegressionModel, RandomForestModel, XGBoostModel,
    GradientBoostingModel, ModelEvaluator, MLModelInterface
)
from app.services.visualizer import PlotlyVisualizer


class FraudDetectionPipeline:
    """
    Pipeline completo para detección de fraude
    Implementa el patrón Pipeline y orquesta todos los componentes
    """
    
    def __init__(self, config):
        """
        Args:
            config: Objeto de configuración
        """
        self.config = config
        
        # Inicializar componentes (Dependency Injection)
        self.data_loader = DataLoader(CSVDataLoader())
        self.preprocessor = DataPreprocessor(
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE
        )
        self.visualizer = PlotlyVisualizer(template=config.PLOTLY_TEMPLATE)
        self.evaluator = ModelEvaluator()
        
        # Estado del pipeline
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.trained_models = []
        self.evaluation_results = []
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Fase 1: Cargar datos
        
        Args:
            file_path: Ruta al archivo de datos
            
        Returns:
            DataFrame con los datos
        """
        print("\n" + "="*60)
        print("FASE 1: CARGA DE DATOS")
        print("="*60)
        
        self.data = self.data_loader.load_data(file_path)
        info = self.data_loader.get_data_info(self.data)
        
        print(f"\nInformación del dataset:")
        print(f"  - Shape: {info['shape']}")
        print(f"  - Memoria: {info['memory_usage']:.2f} MB")
        print(f"  - Duplicados: {info['duplicates']}")
        
        return self.data
    
    def explore_data(self, target_col: str = 'Class') -> Dict[str, Any]:
        """
        Fase 2: Exploración de datos
        
        Args:
            target_col: Nombre de la columna objetivo
            
        Returns:
            Diccionario con información exploratoria
        """
        print("\n" + "="*60)
        print("FASE 2: EXPLORACIÓN DE DATOS")
        print("="*60)
        
        if self.data is None:
            raise ValueError("Debe cargar los datos primero")
        
        # Estadísticas básicas
        stats = {
            'describe': self.data.describe(),
            'class_distribution': self.data[target_col].value_counts(),
            'class_percentage': self.data[target_col].value_counts(normalize=True) * 100
        }
        
        print(f"\nDistribución de clases:")
        for clase, count in stats['class_distribution'].items():
            pct = stats['class_percentage'][clase]
            print(f"  Clase {clase}: {count} ({pct:.2f}%)")
        
        # Crear visualizaciones
        visualizations = {
            'class_distribution': self.visualizer.create_class_distribution(
                self.data[target_col], 
                title="Distribución de Clases (Normal vs Fraude)"
            ),
            'time_analysis': self.visualizer.create_transaction_time_analysis(
                self.data, time_col='Time', class_col=target_col
            ) if 'Time' in self.data.columns else None,
            'correlation_heatmap': self.visualizer.create_correlation_heatmap(
                self.data.corr(), target_col=target_col
            )
        }
        
        return {
            'statistics': stats,
            'visualizations': visualizations
        }
    
    def preprocess_data(self, target_col: str = 'Class', 
                       balance_method: str = 'smote',
                       scale_method: str = 'robust') -> Tuple:
        """
        Fase 3: Preprocesamiento de datos
        
        Args:
            target_col: Nombre de la columna objetivo
            balance_method: Método de balanceo ('smote', 'undersample', 'combined')
            scale_method: Método de escalado ('standard', 'robust')
            
        Returns:
            Tupla con datos procesados
        """
        print("\n" + "="*60)
        print("FASE 3: PREPROCESAMIENTO DE DATOS")
        print("="*60)
        
        if self.data is None:
            raise ValueError("Debe cargar los datos primero")
        
        # Limpiar datos
        print("\n1. Limpieza de datos...")
        cleaned_data = self.preprocessor.clean_data(self.data)
        
        # Dividir datos
        print("\n2. División de datos...")
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(
            cleaned_data, target_col
        )
        
        # Balancear datos de entrenamiento
        print(f"\n3. Balanceo de datos usando {balance_method}...")
        X_train_balanced, y_train_balanced = self.preprocessor.balance_data(
            X_train, y_train, method=balance_method
        )
        
        # Escalar características
        print(f"\n4. Escalado de características usando {scale_method}...")
        self.X_train, self.X_test = self.preprocessor.scale_features(
            X_train_balanced, X_test, method=scale_method
        )
        self.y_train = y_train_balanced
        self.y_test = y_test
        
        print(f"\n✓ Preprocesamiento completado")
        print(f"  Train shape: {self.X_train.shape}")
        print(f"  Test shape: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self, models: List[MLModelInterface] = None):
        """
        Fase 4: Entrenamiento de modelos
        
        Args:
            models: Lista de modelos a entrenar (si es None, usa modelos por defecto)
        """
        print("\n" + "="*60)
        print("FASE 4: ENTRENAMIENTO DE MODELOS")
        print("="*60)
        
        if self.X_train is None:
            raise ValueError("Debe preprocesar los datos primero")
        
        # Usar modelos por defecto si no se especifican
        if models is None:
            models = [
                LogisticRegressionModel(),
                RandomForestModel(n_estimators=100),
                XGBoostModel(n_estimators=100),
                GradientBoostingModel(n_estimators=100)
            ]
        
        self.trained_models = []
        
        for model in models:
            print(f"\nEntrenando {model.get_model_name()}...")
            model.train(self.X_train, self.y_train)
            self.trained_models.append(model)
        
        print(f"\n✓ {len(self.trained_models)} modelos entrenados exitosamente")
    
    def evaluate_models(self) -> List[Dict[str, Any]]:
        """
        Fase 5: Evaluación de modelos
        
        Returns:
            Lista con resultados de evaluación
        """
        print("\n" + "="*60)
        print("FASE 5: EVALUACIÓN DE MODELOS")
        print("="*60)
        
        if not self.trained_models:
            raise ValueError("Debe entrenar los modelos primero")
        
        self.evaluation_results = []
        
        for model in self.trained_models:
            print(f"\nEvaluando {model.get_model_name()}...")
            result = self.evaluator.evaluate_model(model, self.X_test, self.y_test)
            self.evaluation_results.append(result)
            
            # Mostrar métricas principales
            metrics = result['metrics']
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return self.evaluation_results
    
    def create_visualizations(self) -> Dict[str, Any]:
        """
        Fase 6: Crear visualizaciones
        
        Returns:
            Diccionario con todas las visualizaciones
        """
        print("\n" + "="*60)
        print("FASE 6: GENERACIÓN DE VISUALIZACIONES")
        print("="*60)
        
        if not self.evaluation_results:
            raise ValueError("Debe evaluar los modelos primero")
        
        visualizations = {}
        
        # Comparación de modelos
        print("\n1. Creando comparación de modelos...")
        comparison_df = self.evaluator.compare_models(self.evaluation_results)
        visualizations['comparison_table'] = comparison_df
        visualizations['comparison_radar'] = self.visualizer.create_metrics_comparison(comparison_df)
        visualizations['comparison_bars'] = self.visualizer.create_metrics_bar_chart(comparison_df)
        
        # Curvas ROC y Precision-Recall
        print("2. Creando curvas ROC y Precision-Recall...")
        visualizations['roc_curve'] = self.visualizer.create_roc_curve(self.evaluation_results)
        visualizations['pr_curve'] = self.visualizer.create_precision_recall_curve(self.evaluation_results)
        
        # Matriz de confusión del mejor modelo
        print("3. Creando matriz de confusión...")
        best_model_result = max(self.evaluation_results, key=lambda x: x['metrics']['f1_score'])
        visualizations['confusion_matrix'] = self.visualizer.create_confusion_matrix(
            best_model_result['confusion_matrix']
        )
        visualizations['best_model_name'] = best_model_result['model_name']
        
        # Feature importance (si está disponible)
        print("4. Creando importancia de características...")
        for model in self.trained_models:
            if model.get_model_name() == best_model_result['model_name']:
                importance = model.get_feature_importance()
                if importance is not None:
                    visualizations['feature_importance'] = self.visualizer.create_feature_importance(
                        list(self.X_train.columns),
                        importance
                    )
                break
        
        print("\n✓ Visualizaciones generadas exitosamente")
        
        return visualizations
    
    def run_complete_pipeline(self, file_path: str, target_col: str = 'Class',
                             balance_method: str = 'smote',
                             scale_method: str = 'robust') -> Dict[str, Any]:
        """
        Ejecutar el pipeline completo
        
        Args:
            file_path: Ruta al archivo de datos
            target_col: Columna objetivo
            balance_method: Método de balanceo
            scale_method: Método de escalado
            
        Returns:
            Diccionario con todos los resultados
        """
        print("\n" + "#"*60)
        print("# PIPELINE DE DETECCIÓN DE FRAUDE - INICIO")
        print("#"*60)
        
        # Fase 1: Cargar datos
        self.load_data(file_path)
        
        # Fase 2: Explorar datos
        exploration = self.explore_data(target_col)
        
        # Fase 3: Preprocesar
        self.preprocess_data(target_col, balance_method, scale_method)
        
        # Fase 4: Entrenar modelos
        self.train_models()
        
        # Fase 5: Evaluar modelos
        self.evaluate_models()
        
        # Fase 6: Crear visualizaciones
        visualizations = self.create_visualizations()
        
        print("\n" + "#"*60)
        print("# PIPELINE COMPLETADO EXITOSAMENTE")
        print("#"*60)
        
        return {
            'exploration': exploration,
            'evaluation_results': self.evaluation_results,
            'visualizations': visualizations,
            'comparison_df': visualizations['comparison_table']
        }
    
    def save_best_model(self, output_path: str):
        """
        Guardar el mejor modelo
        
        Args:
            output_path: Ruta donde guardar el modelo
        """
        if not self.evaluation_results:
            raise ValueError("Debe evaluar los modelos primero")
        
        # Encontrar el mejor modelo
        best_result = max(self.evaluation_results, key=lambda x: x['metrics']['f1_score'])
        best_model_name = best_result['model_name']
        
        for model in self.trained_models:
            if model.get_model_name() == best_model_name:
                joblib.dump(model, output_path)
                print(f"\n✓ Mejor modelo ({best_model_name}) guardado en {output_path}")
                break