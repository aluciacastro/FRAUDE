# -*- coding: utf-8 -*-
"""
Servicio para cargar datos
Principio SOLID: Single Responsibility - Solo carga datos
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from config.settings import DATASET_PATH, DATA_DIR


class DataLoader:
    """Clase responsable de cargar datos"""

    @staticmethod
    def load_creditcard_dataset(file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Cargar el dataset de detección de fraude

        Args:
            file_path: Ruta al archivo CSV. Si es None, usa la ruta por defecto

        Returns:
            DataFrame con los datos cargados
        """
        if file_path is None:
            file_path = DATASET_PATH

        if not Path(file_path).exists():
            raise FileNotFoundError(
                f"Archivo no encontrado: {file_path}\n"
                f"Por favor, descarga el dataset de Kaggle: "
                f"https://www.kaggle.com/mlg-ulb/creditcardfraud"
            )

        print(f"Cargando dataset desde: {file_path}")
        df = pd.read_csv(file_path)
        print(f" Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")

        return df

    @staticmethod
    def validate_dataset(df: pd.DataFrame) -> dict:
        """
        Validar la estructura y calidad del dataset

        Args:
            df: DataFrame a validar

        Returns:
            Diccionario con información de validación
        """
        validation_info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'missing_values': df.isnull().sum().sum(),
            'duplicates': df.duplicated().sum(),
            'dtypes': df.dtypes.to_dict()
        }

        # Verificar columnas esperadas
        expected_columns = ['Time', 'Amount', 'Class']
        missing_columns = set(expected_columns) - set(df.columns)

        if missing_columns:
            validation_info['warning'] = f"Columnas faltantes: {missing_columns}"
        else:
            validation_info['status'] = 'valid'

        # Verificar distribución de clases
        if 'Class' in df.columns:
            class_distribution = df['Class'].value_counts()
            validation_info['class_distribution'] = {
                'normal': int(class_distribution.get(0, 0)),
                'fraud': int(class_distribution.get(1, 0)),
                'fraud_percentage': float((df['Class'].sum() / len(df)) * 100)
            }

        return validation_info

    @staticmethod
    def get_dataset_summary(df: pd.DataFrame) -> dict:
        """
        Obtener resumen estadístico del dataset

        Args:
            df: DataFrame a resumir

        Returns:
            Diccionario con estadísticas del dataset
        """
        summary = {
            'total_transactions': len(df),
            'features_count': len(df.columns) - 1,  # Excluir 'Class'
            'statistics': df.describe().to_dict()
        }

        if 'Class' in df.columns:
            fraud_transactions = df[df['Class'] == 1]
            normal_transactions = df[df['Class'] == 0]

            summary['fraud_analysis'] = {
                'total_frauds': len(fraud_transactions),
                'fraud_percentage': (len(fraud_transactions) / len(df)) * 100,
                'imbalance_ratio': f"1:{int(len(normal_transactions) / len(fraud_transactions))}"
            }

            if 'Amount' in df.columns:
                summary['amount_analysis'] = {
                    'fraud': {
                        'mean': float(fraud_transactions['Amount'].mean()),
                        'median': float(fraud_transactions['Amount'].median()),
                        'max': float(fraud_transactions['Amount'].max())
                    },
                    'normal': {
                        'mean': float(normal_transactions['Amount'].mean()),
                        'median': float(normal_transactions['Amount'].median()),
                        'max': float(normal_transactions['Amount'].max())
                    }
                }

        return summary

    @staticmethod
    def load_and_validate(file_path: Optional[Path] = None) -> Tuple[pd.DataFrame, dict]:
        """
        Cargar y validar el dataset en un solo paso

        Args:
            file_path: Ruta al archivo CSV

        Returns:
            Tupla con (DataFrame, información de validación)
        """
        df = DataLoader.load_creditcard_dataset(file_path)
        validation_info = DataLoader.validate_dataset(df)

        return df, validation_info


class DataSplitter:
    """Clase responsable de dividir datos en conjuntos de entrenamiento y prueba"""

    @staticmethod
    def split_data(
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify_column: Optional[str] = 'Class'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Dividir datos en entrenamiento y prueba

        Args:
            df: DataFrame completo
            test_size: Proporción para conjunto de prueba
            random_state: Semilla aleatoria
            stratify_column: Columna para estratificación

        Returns:
            Tupla con (df_train, df_test)
        """
        from sklearn.model_selection import train_test_split

        stratify = df[stratify_column] if stratify_column and stratify_column in df.columns else None

        df_train, df_test = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )

        print(f"Dataset dividido:")
        print(f"  Entrenamiento: {df_train.shape}")
        print(f"  Prueba: {df_test.shape}")

        if stratify_column and stratify_column in df.columns:
            print(f"\nDistribución de {stratify_column}:")
            print(f"  Entrenamiento: {df_train[stratify_column].value_counts().to_dict()}")
            print(f"  Prueba: {df_test[stratify_column].value_counts().to_dict()}")

        return df_train, df_test
    
# -*- coding: utf-8 -*-
# ... (tu código actual) ...

class CSVDataLoader:
    """Clase para cargar datos desde archivos CSV personalizados"""
    
    @staticmethod
    def load_from_file(file_path: str) -> pd.DataFrame:
        """
        Cargar datos desde un archivo CSV
        
        Args:
            file_path: Ruta al archivo CSV
            
        Returns:
            DataFrame con los datos cargados
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        df = pd.read_csv(file_path)
        print(f"Archivo cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
        
        return df
    
    @staticmethod
    def load_from_upload(file_storage) -> pd.DataFrame:
        """
        Cargar datos desde un archivo subido (Flask FileStorage)
        
        Args:
            file_storage: Objeto FileStorage de Flask
            
        Returns:
            DataFrame con los datos cargados
        """
        df = pd.read_csv(file_storage)
        print(f"Archivo subido cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
        
        return df