"""
Servicio para preprocesamiento de datos
Principio SOLID: Single Responsibility - Solo preprocesa datos
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from typing import Tuple, Optional


class DataPreprocessor:
    """Clase responsable del preprocesamiento de datos"""

    def __init__(self):
        self.scaler = RobustScaler()
        self.is_fitted = False

    def scale_features(
        self,
        df: pd.DataFrame,
        features_to_scale: list = ['Time', 'Amount']
    ) -> pd.DataFrame:
        """
        Escalar caracter�sticas usando RobustScaler

        Args:
            df: DataFrame con datos
            features_to_scale: Lista de columnas a escalar

        Returns:
            DataFrame con caracter�sticas escaladas
        """
        df_processed = df.copy()

        for feature in features_to_scale:
            if feature in df_processed.columns:
                scaled_feature_name = f"{feature}_scaled"
                df_processed[scaled_feature_name] = self.scaler.fit_transform(
                    df_processed[[feature]]
                )
                # Eliminar la columna original
                df_processed.drop(feature, axis=1, inplace=True)
                print(f" Feature '{feature}' escalada y renombrada a '{scaled_feature_name}'")

        self.is_fitted = True
        return df_processed

    def prepare_features_target(
        self,
        df: pd.DataFrame,
        target_column: str = 'Class'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separar caracter�sticas (X) y variable objetivo (y)

        Args:
            df: DataFrame completo
            target_column: Nombre de la columna objetivo

        Returns:
            Tupla (X, y)
        """
        if target_column not in df.columns:
            raise ValueError(f"Columna '{target_column}' no encontrada en el DataFrame")

        X = df.drop(target_column, axis=1)
        y = df[target_column]

        print(f"Features (X): {X.shape}")
        print(f"Target (y): {y.shape}")
        print(f"Distribuci�n del target: {y.value_counts().to_dict()}")

        return X, y

    @staticmethod
    def get_correlation_with_target(
        df: pd.DataFrame,
        target_column: str = 'Class',
        top_n: int = 15
    ) -> pd.Series:
        """
        Obtener correlaciones con la variable objetivo

        Args:
            df: DataFrame con datos
            target_column: Columna objetivo
            top_n: N�mero de correlaciones m�s altas a retornar

        Returns:
            Series con correlaciones ordenadas
        """
        correlations = df.corr()[target_column].abs().sort_values(ascending=False)
        return correlations.head(top_n + 1)[1:]  # Excluir la correlaci�n consigo misma


class BalancingStrategy:
    """Clase base para estrategias de balanceo"""

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name

    def balance(self, X, y) -> Tuple:
        """M�todo a implementar por las estrategias concretas"""
        raise NotImplementedError


class NoBalancingStrategy(BalancingStrategy):
    """Sin balanceo de clases"""

    def __init__(self):
        super().__init__('Original')

    def balance(self, X, y) -> Tuple:
        """Retorna los datos sin modificaci�n"""
        print(f"Estrategia: {self.strategy_name} (sin balanceo)")
        print(f"  Distribuci�n: {pd.Series(y).value_counts().to_dict()}")
        return X, y


class SMOTEBalancingStrategy(BalancingStrategy):
    """Balanceo usando SMOTE (Synthetic Minority Over-sampling Technique)"""

    def __init__(self, sampling_strategy: float = 0.5):
        super().__init__('SMOTE')
        self.sampling_strategy = sampling_strategy
        self.smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy)

    def balance(self, X, y) -> Tuple:
        """Aplica SMOTE para balancear las clases"""
        print(f"Estrategia: {self.strategy_name}")
        print(f"  Distribuci�n original: {pd.Series(y).value_counts().to_dict()}")

        X_balanced, y_balanced = self.smote.fit_resample(X, y)

        print(f"  Distribuci�n balanceada: {pd.Series(y_balanced).value_counts().to_dict()}")
        return X_balanced, y_balanced


class UnderSamplingStrategy(BalancingStrategy):
    """Balanceo usando Random Under-sampling"""

    def __init__(self, sampling_strategy: float = 0.5):
        super().__init__('Under-sampling')
        self.sampling_strategy = sampling_strategy
        self.rus = RandomUnderSampler(random_state=42, sampling_strategy=sampling_strategy)

    def balance(self, X, y) -> Tuple:
        """Aplica under-sampling para balancear las clases"""
        print(f"Estrategia: {self.strategy_name}")
        print(f"  Distribuci�n original: {pd.Series(y).value_counts().to_dict()}")

        X_balanced, y_balanced = self.rus.fit_resample(X, y)

        print(f"  Distribuci�n balanceada: {pd.Series(y_balanced).value_counts().to_dict()}")
        return X_balanced, y_balanced


class BalancingStrategyFactory:
    """Factory para crear estrategias de balanceo"""

    @staticmethod
    def create_strategy(strategy_name: str, **kwargs) -> BalancingStrategy:
        """
        Crear una estrategia de balanceo

        Args:
            strategy_name: Nombre de la estrategia ('Original', 'SMOTE', 'Under-sampling')
            **kwargs: Par�metros adicionales para la estrategia

        Returns:
            Instancia de la estrategia
        """
        strategies = {
            'Original': NoBalancingStrategy,
            'SMOTE': SMOTEBalancingStrategy,
            'Under-sampling': UnderSamplingStrategy
        }

        if strategy_name not in strategies:
            raise ValueError(f"Estrategia desconocida: {strategy_name}")

        if strategy_name == 'Original':
            return strategies[strategy_name]()
        else:
            return strategies[strategy_name](**kwargs)

    @staticmethod
    def create_all_strategies() -> dict:
        """Crear todas las estrategias disponibles"""
        return {
            'Original': NoBalancingStrategy(),
            'SMOTE': SMOTEBalancingStrategy(sampling_strategy=0.5),
            'Under-sampling': UnderSamplingStrategy(sampling_strategy=0.5)
        }


class DataProcessor:
    """Clase de alto nivel para procesamiento completo de datos"""

    def __init__(self):
        self.preprocessor = DataPreprocessor()

    def process_data(
        self,
        df: pd.DataFrame,
        features_to_scale: list = ['Time', 'Amount'],
        target_column: str = 'Class'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Procesamiento completo de datos

        Args:
            df: DataFrame original
            features_to_scale: Caracter�sticas a escalar
            target_column: Columna objetivo

        Returns:
            Tupla (X, y) con datos procesados
        """
        print("="*60)
        print("PROCESAMIENTO DE DATOS")
        print("="*60)

        # Escalar caracter�sticas
        df_processed = self.preprocessor.scale_features(df, features_to_scale)

        # Separar X y y
        X, y = self.preprocessor.prepare_features_target(df_processed, target_column)

        print(" Procesamiento completado")
        return X, y

    def process_and_balance(
        self,
        df: pd.DataFrame,
        strategy_name: str = 'SMOTE',
        features_to_scale: list = ['Time', 'Amount'],
        target_column: str = 'Class'
    ) -> Tuple:
        """
        Procesar datos y aplicar estrategia de balanceo

        Args:
            df: DataFrame original
            strategy_name: Nombre de la estrategia de balanceo
            features_to_scale: Caracter�sticas a escalar
            target_column: Columna objetivo

        Returns:
            Tupla (X_balanced, y_balanced)
        """
        # Procesar datos
        X, y = self.process_data(df, features_to_scale, target_column)

        # Aplicar balanceo
        print(f"\n{'='*60}")
        print(f"BALANCEO DE CLASES")
        print(f"{'='*60}")

        strategy = BalancingStrategyFactory.create_strategy(strategy_name)
        X_balanced, y_balanced = strategy.balance(X, y)

        return X_balanced, y_balanced
