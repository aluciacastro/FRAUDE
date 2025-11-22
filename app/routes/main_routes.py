# -*- coding: utf-8 -*-
"""
Rutas de la aplicación web
"""
from flask import Blueprint, render_template, request, jsonify, current_app, session
import pandas as pd
import os
from werkzeug.utils import secure_filename
from pathlib import Path

from app.pipelines.fraud_pipeline import FraudDetectionPipeline
from config import settings as Config


bp = Blueprint('main', __name__)

# Variable global para el pipeline (en producción usar Redis o similar)
pipeline_instance = None


@bp.route('/')
def index():
    """Página principal"""
    return render_template('index.html')


@bp.route('/upload', methods=['POST'])
def upload_file():
    """
    Endpoint para subir archivo de datos
    """
    global pipeline_instance
    
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No se encontró archivo'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No se seleccionó archivo'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'success': False, 'error': 'El archivo debe ser un CSV'}), 400
        
        # Guardar archivo
        filename = secure_filename(file.filename)
        
        # Asegurar que existe el directorio data
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        
        filepath = data_dir / filename
        file.save(str(filepath))
        
        # Inicializar pipeline
        pipeline_instance = FraudDetectionPipeline(Config)
        
        # Cargar datos
        data = pipeline_instance.load_data(str(filepath))
        
        # Obtener información básica
        info = {
            'rows': len(data),
            'columns': len(data.columns),
            'size_mb': round(data.memory_usage(deep=True).sum() / 1024**2, 2)
        }
        
        session['data_loaded'] = True
        session['filename'] = filename
        
        return jsonify({
            'success': True,
            'message': 'Archivo cargado exitosamente',
            'info': info
        })
        
    except Exception as e:
        print(f"Error en upload: {str(e)}")  # Para debug
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/analyze', methods=['POST'])
def analyze():
    """
    Ejecutar análisis exploratorio
    """
    global pipeline_instance
    
    try:
        if pipeline_instance is None or pipeline_instance.data is None:
            return jsonify({'success': False, 'error': 'Debe cargar datos primero'}), 400
        
        # Obtener parámetros
        data = request.get_json()
        target_col = data.get('target_col', 'Class')
        
        # Explorar datos
        exploration = pipeline_instance.explore_data(target_col)
        
        # Convertir visualizaciones a JSON
        visualizations = {}
        for key, fig in exploration['visualizations'].items():
            if fig is not None:
                visualizations[key] = fig.to_json()
        
        # Estadísticas
        stats = exploration['statistics']
        class_dist = stats['class_distribution'].to_dict()
        class_pct = stats['class_percentage'].to_dict()
        
        return jsonify({
            'success': True,
            'visualizations': visualizations,
            'statistics': {
                'class_distribution': class_dist,
                'class_percentage': {k: round(v, 2) for k, v in class_pct.items()}
            }
        })
        
    except Exception as e:
        print(f"Error en analyze: {str(e)}")  # Para debug
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/train', methods=['POST'])
def train():
    """
    Entrenar modelos
    """
    global pipeline_instance
    
    try:
        if pipeline_instance is None or pipeline_instance.data is None:
            return jsonify({'success': False, 'error': 'Debe cargar datos primero'}), 400
        
        # Obtener parámetros
        data = request.get_json()
        target_col = data.get('target_col', 'Class')
        balance_method = data.get('balance_method', 'smote')
        scale_method = data.get('scale_method', 'robust')
        
        # Preprocesar
        pipeline_instance.preprocess_data(target_col, balance_method, scale_method)
        
        # Entrenar modelos
        pipeline_instance.train_models()
        
        # Evaluar modelos
        pipeline_instance.evaluate_models()
        
        # Crear visualizaciones
        visualizations = pipeline_instance.create_visualizations()
        
        # Convertir visualizaciones a JSON
        viz_json = {}
        for key, value in visualizations.items():
            if key == 'comparison_table':
                viz_json[key] = value.to_dict('records')
            elif key == 'best_model_name':
                viz_json[key] = value
            elif hasattr(value, 'to_json'):
                viz_json[key] = value.to_json()
        
        return jsonify({
            'success': True,
            'message': 'Modelos entrenados exitosamente',
            'visualizations': viz_json,
            'best_model': visualizations['best_model_name']
        })
        
    except Exception as e:
        print(f"Error en train: {str(e)}")  # Para debug
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/results')
def results():
    """
    Página de resultados
    """
    return render_template('results.html')


@bp.route('/about')
def about():
    """
    Página sobre CRISP-DM
    """
    return render_template('about.html')


@bp.route('/analysis')
def analysis():
    """
    Página de análisis exploratorio
    """
    return render_template('analysis.html')


@bp.route('/api/status')
def status():
    """
    Estado del pipeline
    """
    global pipeline_instance
    
    status_info = {
        'data_loaded': pipeline_instance is not None and pipeline_instance.data is not None,
        'data_preprocessed': pipeline_instance is not None and hasattr(pipeline_instance, 'X_train') and pipeline_instance.X_train is not None,
        'models_trained': pipeline_instance is not None and hasattr(pipeline_instance, 'trained_models') and len(pipeline_instance.trained_models) > 0,
        'models_evaluated': pipeline_instance is not None and hasattr(pipeline_instance, 'evaluation_results') and len(pipeline_instance.evaluation_results) > 0
    }
    
    return jsonify(status_info)


@bp.route('/download-sample')
def download_sample():
    """
    Información sobre dataset de ejemplo
    """
    return jsonify({
        'dataset_name': 'Credit Card Fraud Detection',
        'kaggle_url': 'https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud',
        'description': 'Dataset de detección de fraude en tarjetas de crédito',
        'features': 31,
        'rows': '~284,807'
    })