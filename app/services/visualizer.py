"""
Servicio de visualización con Plotly
Principio SOLID: Single Responsibility - Solo crea visualizaciones
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List


class PlotlyVisualizer:
    """
    Clase para crear visualizaciones interactivas con Plotly
    """
    
    def __init__(self, template: str = 'plotly_white'):
        """
        Args:
            template: Template de Plotly a usar
        """
        self.template = template
        self.default_height = 500
        self.colors = px.colors.qualitative.Set2
    
    def create_class_distribution(self, y: pd.Series, title: str = "Distribución de Clases") -> go.Figure:
        """
        Crear gráfico de distribución de clases
        
        Args:
            y: Serie con las clases
            title: Título del gráfico
            
        Returns:
            Figura de Plotly
        """
        value_counts = y.value_counts()
        percentages = (value_counts / len(y) * 100).round(2)
        
        fig = go.Figure(data=[
            go.Bar(
                x=value_counts.index,
                y=value_counts.values,
                text=[f'{count}<br>({pct}%)' for count, pct in zip(value_counts.values, percentages.values)],
                textposition='auto',
                marker_color=self.colors[:len(value_counts)]
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Clase",
            yaxis_title="Cantidad",
            template=self.template,
            height=self.default_height,
            showlegend=False
        )
        
        return fig
    
    def create_confusion_matrix(self, cm: np.ndarray, labels: List[str] = None) -> go.Figure:
        """
        Crear matriz de confusión interactiva
        
        Args:
            cm: Matriz de confusión
            labels: Etiquetas de las clases
            
        Returns:
            Figura de Plotly
        """
        if labels is None:
            labels = ['Normal', 'Fraude']
        
        # Calcular porcentajes
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Crear texto para cada celda
        text = [[f'{cm[i][j]}<br>({cm_percent[i][j]:.1f}%)' 
                for j in range(len(cm[0]))] 
                for i in range(len(cm))]
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            text=text,
            texttemplate='%{text}',
            textfont={"size": 16},
            colorscale='Blues',
            showscale=True
        ))
        
        fig.update_layout(
            title='Matriz de Confusión',
            xaxis_title='Predicción',
            yaxis_title='Valor Real',
            template=self.template,
            height=self.default_height
        )
        
        return fig
    
    def create_roc_curve(self, results_list: List[Dict]) -> go.Figure:
        """
        Crear curva ROC para múltiples modelos
        
        Args:
            results_list: Lista de resultados de evaluación
            
        Returns:
            Figura de Plotly
        """
        fig = go.Figure()
        
        for i, result in enumerate(results_list):
            fpr = result['roc_curve']['fpr']
            tpr = result['roc_curve']['tpr']
            roc_auc = result['metrics']['roc_auc']
            model_name = result['model_name']
            
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'{model_name} (AUC = {roc_auc:.3f})',
                line=dict(color=self.colors[i % len(self.colors)], width=2)
            ))
        
        # Línea diagonal (clasificador aleatorio)
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Aleatorio (AUC = 0.500)',
            line=dict(color='gray', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Curva ROC - Comparación de Modelos',
            xaxis_title='Tasa de Falsos Positivos (FPR)',
            yaxis_title='Tasa de Verdaderos Positivos (TPR)',
            template=self.template,
            height=self.default_height,
            legend=dict(x=0.6, y=0.1)
        )
        
        return fig
    
    def create_precision_recall_curve(self, results_list: List[Dict]) -> go.Figure:
        """
        Crear curva Precision-Recall para múltiples modelos
        
        Args:
            results_list: Lista de resultados de evaluación
            
        Returns:
            Figura de Plotly
        """
        fig = go.Figure()
        
        for i, result in enumerate(results_list):
            precision = result['precision_recall_curve']['precision']
            recall = result['precision_recall_curve']['recall']
            model_name = result['model_name']
            f1 = result['metrics']['f1_score']
            
            fig.add_trace(go.Scatter(
                x=recall,
                y=precision,
                mode='lines',
                name=f'{model_name} (F1 = {f1:.3f})',
                line=dict(color=self.colors[i % len(self.colors)], width=2)
            ))
        
        fig.update_layout(
            title='Curva Precision-Recall - Comparación de Modelos',
            xaxis_title='Recall',
            yaxis_title='Precision',
            template=self.template,
            height=self.default_height,
            legend=dict(x=0.6, y=0.9)
        )
        
        return fig
    
    def create_feature_importance(self, feature_names: List[str], 
                                  importances: np.ndarray, 
                                  top_n: int = 15) -> go.Figure:
        """
        Crear gráfico de importancia de características
        
        Args:
            feature_names: Nombres de las características
            importances: Valores de importancia
            top_n: Número de características a mostrar
            
        Returns:
            Figura de Plotly
        """
        # Crear DataFrame y ordenar
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True).tail(top_n)
        
        fig = go.Figure(go.Bar(
            x=df_importance['importance'],
            y=df_importance['feature'],
            orientation='h',
            marker_color=self.colors[0]
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Características Más Importantes',
            xaxis_title='Importancia',
            yaxis_title='Característica',
            template=self.template,
            height=max(self.default_height, top_n * 25)
        )
        
        return fig
    
    def create_metrics_comparison(self, comparison_df: pd.DataFrame) -> go.Figure:
        """
        Crear gráfico de comparación de métricas entre modelos
        
        Args:
            comparison_df: DataFrame con métricas de modelos
            
        Returns:
            Figura de Plotly
        """
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        fig = go.Figure()
        
        for i, model in enumerate(comparison_df['Model']):
            fig.add_trace(go.Scatterpolar(
                r=[comparison_df.loc[i, metric] for metric in metrics],
                theta=metrics,
                fill='toself',
                name=model,
                line_color=self.colors[i % len(self.colors)]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title='Comparación de Métricas entre Modelos',
            template=self.template,
            height=600
        )
        
        return fig
    
    def create_metrics_bar_chart(self, comparison_df: pd.DataFrame) -> go.Figure:
        """
        Crear gráfico de barras de métricas
        
        Args:
            comparison_df: DataFrame con métricas de modelos
            
        Returns:
            Figura de Plotly
        """
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        fig = go.Figure()
        
        for metric in metrics:
            fig.add_trace(go.Bar(
                name=metric,
                x=comparison_df['Model'],
                y=comparison_df[metric],
                text=comparison_df[metric].round(3),
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Comparación de Métricas por Modelo',
            xaxis_title='Modelo',
            yaxis_title='Score',
            barmode='group',
            template=self.template,
            height=self.default_height,
            yaxis=dict(range=[0, 1.05])
        )
        
        return fig
    
    def create_correlation_heatmap(self, correlation_df: pd.DataFrame, 
                                   target_col: str = 'Class') -> go.Figure:
        """
        Crear mapa de calor de correlaciones
        
        Args:
            correlation_df: DataFrame con correlaciones
            target_col: Nombre de la columna objetivo
            
        Returns:
            Figura de Plotly
        """
        # Obtener correlaciones con el target
        target_corr = correlation_df[target_col].sort_values(ascending=False)
        
        # Tomar top features
        top_features = target_corr.head(16).index.tolist()
        corr_matrix = correlation_df.loc[top_features, top_features]
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlación")
        ))
        
        fig.update_layout(
            title='Mapa de Correlación - Top Features',
            template=self.template,
            height=700,
            width=800
        )
        
        return fig
    
    def create_transaction_time_analysis(self, df: pd.DataFrame, 
                                        time_col: str = 'Time',
                                        class_col: str = 'Class') -> go.Figure:
        """
        Análisis de transacciones en el tiempo
        
        Args:
            df: DataFrame con los datos
            time_col: Columna de tiempo
            class_col: Columna de clase
            
        Returns:
            Figura de Plotly
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Transacciones Normales', 'Transacciones Fraudulentas'),
            vertical_spacing=0.15
        )
        
        # Transacciones normales
        normal = df[df[class_col] == 0][time_col]
        fig.add_trace(
            go.Histogram(x=normal, name='Normal', marker_color=self.colors[0], nbinsx=50),
            row=1, col=1
        )
        
        # Transacciones fraudulentas
        fraud = df[df[class_col] == 1][time_col]
        fig.add_trace(
            go.Histogram(x=fraud, name='Fraude', marker_color=self.colors[1], nbinsx=50),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Tiempo", row=2, col=1)
        fig.update_yaxes(title_text="Frecuencia", row=1, col=1)
        fig.update_yaxes(title_text="Frecuencia", row=2, col=1)
        
        fig.update_layout(
            title='Distribución Temporal de Transacciones',
            template=self.template,
            height=700,
            showlegend=True
        )
        
        return fig