# -*- coding: utf-8 -*-
"""
Procesador del Notebook para generar datos precargados
Este script ejecuta el notebook y guarda los resultados
"""
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from app.pipelines.fraud_pipeline import FraudDetectionPipeline
from config import settings


class NotebookDataGenerator:
    """Genera datos desde el notebook para la aplicación web"""
    
    def __init__(self):
        self.data_dir = Path('data')
        self.results_dir = Path('data/precomputed')
        self.results_dir.mkdir(exist_ok=True)
        self.dataset_path = self.data_dir / 'creditcard.csv'
        
    def load_and_analyze_dataset(self):
        """Cargar dataset y generar análisis exploratorio"""
        print("="*60)
        print("CARGANDO Y ANALIZANDO DATASET")
        print("="*60)
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset no encontrado en {self.dataset_path}\n"
                "Descárgalo de: https://www.kaggle.com/mlg-ulb/creditcardfraud"
            )
        
        # Cargar dataset
        df = pd.read_csv(self.dataset_path)
        print(f"✓ Dataset cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")
        
        # Análisis básico
        analysis = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': int(df.duplicated().sum()),
            'memory_mb': float(df.memory_usage(deep=True).sum() / 1024**2)
        }
        
        # Análisis de clases
        class_dist = df['Class'].value_counts().to_dict()
        class_pct = (df['Class'].value_counts(normalize=True) * 100).to_dict()
        
        analysis['class_distribution'] = {
            'counts': {str(k): int(v) for k, v in class_dist.items()},
            'percentages': {str(k): float(v) for k, v in class_pct.items()},
            'fraud_count': int(class_dist.get(1, 0)),
            'normal_count': int(class_dist.get(0, 0)),
            'fraud_percentage': float(class_pct.get(1, 0)),
            'imbalance_ratio': f"1:{int(class_dist[0]/class_dist[1])}"
        }
        
        # Estadísticas de Amount por clase
        analysis['amount_stats'] = {
            'normal': df[df['Class'] == 0]['Amount'].describe().to_dict(),
            'fraud': df[df['Class'] == 1]['Amount'].describe().to_dict()
        }
        
        # Estadísticas de Time
        analysis['time_stats'] = df['Time'].describe().to_dict()
        
        # Correlaciones con Class
        correlations = df.corr()['Class'].sort_values(ascending=False)
        analysis['correlations_with_class'] = {
            'top_positive': correlations.head(11)[1:].to_dict(),  # Top 10 (sin Class)
            'top_negative': correlations.tail(10).to_dict()
        }
        
        # Guardar análisis
        output_path = self.results_dir / 'phase1_data_understanding.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Análisis guardado en {output_path}")
        return analysis
    
    def run_pipeline_and_save_results(self):
        """Ejecutar pipeline completo y guardar resultados"""
        print("\n" + "="*60)
        print("EJECUTANDO PIPELINE COMPLETO")
        print("="*60)
        
        from config.settings import Config
        
        # Crear pipeline
        pipeline = FraudDetectionPipeline(Config)
        
        # Ejecutar pipeline completo
        results = pipeline.run_complete_pipeline(
            file_path=str(self.dataset_path),
            target_col='Class',
            balance_method='smote',
            scale_method='robust'
        )
        
        # Extraer datos para cada fase
        phase_data = {}
        
        # FASE 2: Exploración
        phase_data['phase2_exploration'] = {
            'statistics': results['exploration']['statistics'],
            'class_distribution': results['exploration']['statistics']['class_distribution']
        }
        
        # FASE 3: Preparación (información del preprocesamiento)
        phase_data['phase3_preparation'] = {
            'balance_method': 'smote',
            'scale_method': 'robust',
            'train_shape': pipeline.X_train.shape,
            'test_shape': pipeline.X_test.shape,
            'train_class_dist': pd.Series(pipeline.y_train).value_counts().to_dict(),
            'test_class_dist': pd.Series(pipeline.y_test).value_counts().to_dict()
        }
        
        # FASE 4: Modelado (información de modelos)
        phase_data['phase4_modeling'] = {
            'models_trained': [m.get_model_name() for m in pipeline.trained_models],
            'total_models': len(pipeline.trained_models)
        }
        
        # FASE 5: Evaluación (resultados completos)
        comparison_df = results['comparison_df']
        phase_data['phase5_evaluation'] = {
            'comparison_table': comparison_df.to_dict('records'),
            'best_model': comparison_df.iloc[0]['Model'],
            'best_metrics': {
                'accuracy': float(comparison_df.iloc[0]['Accuracy']),
                'precision': float(comparison_df.iloc[0]['Precision']),
                'recall': float(comparison_df.iloc[0]['Recall']),
                'f1_score': float(comparison_df.iloc[0]['F1-Score']),
                'roc_auc': float(comparison_df.iloc[0]['ROC-AUC'])
            }
        }
        
        # Guardar cada fase
        for phase_name, data in phase_data.items():
            output_path = self.results_dir / f'{phase_name}.json'
            with open(output_path, 'w', encoding='utf-8') as f:
                # Convertir numpy types a Python types
                json.dump(data, f, indent=2, ensure_ascii=False, default=self._convert_to_serializable)
            print(f"✓ {phase_name} guardado en {output_path}")
        
        # Guardar visualizaciones
        viz_path = self.results_dir / 'visualizations.pkl'
        with open(viz_path, 'wb') as f:
            pickle.dump(results['visualizations'], f)
        print(f"✓ Visualizaciones guardadas en {viz_path}")
        
        # Guardar el mejor modelo
        pipeline.save_best_model(str(self.results_dir / 'best_model.pkl'))
        
        return phase_data
    
    def _convert_to_serializable(self, obj):
        """Convertir objetos numpy a tipos serializables"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)
    
    def generate_all(self):
        """Generar todos los datos precalculados"""
        print("\n" + "#"*60)
        print("# GENERADOR DE DATOS DESDE NOTEBOOK")
        print("#"*60)
        
        try:
            # Fase 1: Análisis exploratorio
            analysis = self.load_and_analyze_dataset()
            
            # Ejecutar pipeline completo
            results = self.run_pipeline_and_save_results()
            
            print("\n" + "#"*60)
            print("# GENERACIÓN COMPLETADA EXITOSAMENTE")
            print("#"*60)
            print(f"\nArchivos generados en: {self.results_dir}")
            
            return True
            
        except Exception as e:
            print(f"\n✗ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == '__main__':
    generator = NotebookDataGenerator()
    success = generator.generate_all()
    
    if success:
        print("\n✓ Ahora puedes ejecutar la aplicación Flask con 'python run.py'")
    else:
        print("\n✗ La generación falló. Revisa los errores arriba.")