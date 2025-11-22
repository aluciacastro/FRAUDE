# -*- coding: utf-8 -*-
"""
Servicio para preprocesamiento de datos
Principio SOLID: Single Responsibility - Solo preprocesa datos
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from typing import Tuple


class DataPreprocessor:
    """
    Clase responsable del preprocesamiento completo de datos
    Principio SOLID: Single Responsibility
    """
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Args:
            test_size: Proporción de datos para test
            random_state: Semilla para reproducibilidad
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = None
        self.is_fitted = False
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpiar datos: eliminar duplicados y valores nulos
        
        Args:
            df: DataFrame original
            
        Returns:
            DataFrame limpio
        """
        df_clean = df.copy()
        
        # Eliminar duplicados
        duplicates_count = df_clean.duplicated().sum()
        if duplicates_count > 0:
            print(f"  Eliminando {duplicates_count} filas duplicadas...")
            df_clean = df_clean.drop_duplicates()
        
        # Eliminar valores nulos
        null_count = df_clean.isnull().sum().sum()
        if null_count > 0:
            print(f"  Eliminando {null_count} valores nulos...")
            df_clean = df_clean.dropna()
        
        print(f"  Dataset limpio: {df_clean.shape}")
        return df_clean
    
    def split_data(self, df: pd.DataFrame, target_col: str = 'Class') -> Tuple:
        """
        Dividir datos en entrenamiento y prueba
        
        Args:
            df: DataFrame completo
            target_col: Nombre de la columna objetivo
            
        Returns:
            Tupla (X_train, X_test, y_train, y_test)
        """
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"  Train class dist: {pd.Series(y_train).value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def balance_data(self, X: pd.DataFrame, y: pd.Series, 
                    method: str = 'smote') -> Tuple:
        """
        Balancear clases usando diferentes técnicas
        
        Args:
            X: Features
            y: Target
            method: 'smote', 'undersample', 'combined', o 'none'
            
        Returns:
            Tupla (X_balanced, y_balanced)
        """
        print(f"  Distribución original: {pd.Series(y).value_counts().to_dict()}")
        
        if method == 'none':
            return X, y
        
        elif method == 'smote':
            sampler = SMOTE(random_state=self.random_state, sampling_strategy=0.5)
            X_balanced, y_balanced = sampler.fit_resample(X, y)
            
        elif method == 'undersample':
            sampler = RandomUnderSampler(random_state=self.random_state, sampling_strategy=0.5)
            X_balanced, y_balanced = sampler.fit_resample(X, y)
            
        elif method == 'combined':
            sampler = SMOTETomek(random_state=self.random_state)
            X_balanced, y_balanced = sampler.fit_resample(X, y)
            
        else:
            raise ValueError(f"Método de balanceo desconocido: {method}")
        
        print(f"  Distribución balanceada: {pd.Series(y_balanced).value_counts().to_dict()}")
        return X_balanced, y_balanced
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                      method: str = 'robust') -> Tuple:
        """
        Escalar características numéricas
        
        Args:
            X_train: Features de entrenamiento
            X_test: Features de prueba
            method: 'standard' o 'robust'
            
        Returns:
            Tupla (X_train_scaled, X_test_scaled)
        """
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Método de escalado desconocido: {method}")
        
        # Escalar solo si hay columnas Time o Amount
        cols_to_scale = []
        if 'Time' in X_train.columns:
            cols_to_scale.append('Time')
        if 'Amount' in X_train.columns:
            cols_to_scale.append('Amount')
        
        if not cols_to_scale:
            print("  No hay columnas Time/Amount para escalar")
            return X_train, X_test
        
        # Crear copias
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        # Escalar
        X_train_scaled[cols_to_scale] = self.scaler.fit_transform(X_train[cols_to_scale])
        X_test_scaled[cols_to_scale] = self.scaler.transform(X_test[cols_to_scale])
        
        print(f"  Columnas escaladas: {cols_to_scale}")
        self.is_fitted = True
        
        return X_train_scaled, X_test_scaled
    
    def get_feature_importance_data(self, X: pd.DataFrame, y: pd.Series,
                                   top_n: int = 15) -> pd.DataFrame:
        """
        Calcular correlaciones con el target
        
        Args:
            X: Features
            y: Target
            top_n: Número de features más importantes
            
        Returns:
            DataFrame con correlaciones
        """
        df_temp = X.copy()
        df_temp['target'] = y
        
        correlations = df_temp.corr()['target'].abs().sort_values(ascending=False)
        correlations = correlations.drop('target')
        
        return correlations.head(top_n)


class DataLoader:
    """Clase para cargar diferentes fuentes de datos"""
    
    def __init__(self, loader_strategy):
        """
        Args:
            loader_strategy: Estrategia de carga (CSVDataLoader, etc.)
        """
        self.loader = loader_strategy
    
    def load_data(self, source) -> pd.DataFrame:
        """
        Cargar datos desde una fuente
        
        Args:
            source: Ruta del archivo o fuente de datos
            
        Returns:
            DataFrame con los datos
        """
        return self.loader.load_from_file(source)
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Obtener información del dataset
        
        Args:
            df: DataFrame
            
        Returns:
            Diccionario con información
        """
        return {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,
            'duplicates': int(df.duplicated().sum()),
            'missing_values': df.isnull().sum().sum()
        }


class CSVDataLoader:
    """Cargador de archivos CSV"""
    
    @staticmethod
    def load_from_file(file_path: str) -> pd.DataFrame:
        """
        Cargar CSV desde archivo
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            DataFrame
        """
        import os
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        df = pd.read_csv(file_path)
        print(f"Archivo cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")
        return df