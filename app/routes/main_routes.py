# -*- coding: utf-8 -*-
"""
Rutas de la aplicación web - Con datos precargados
"""
from flask import Blueprint, render_template, jsonify
import json
import pickle
from pathlib import Path

bp = Blueprint('main', __name__)

# Directorio de datos precalculados
RESULTS_DIR = Path('data/precomputed')


def load_precomputed_data(filename):
    """Cargar datos precalculados desde JSON"""
    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_visualizations():
    """Cargar visualizaciones desde pickle"""
    viz_path = RESULTS_DIR / 'visualizations.pkl'
    if not viz_path.exists():
        return None
    
    with open(viz_path, 'rb') as f:
        return pickle.load(f)


@bp.route('/')
def index():
    """Página principal con fases CRISP-DM"""
    # Cargar datos de la Fase 1
    phase1_data = load_precomputed_data('phase1_data_understanding.json')
    
    if not phase1_data:
        return render_template('index.html', 
                             data_available=False,
                             error_message="Datos no generados. Ejecuta 'python notebook_processor.py' primero.")
    
    return render_template('index.html', 
                         data_available=True,
                         dataset_info=phase1_data)


@bp.route('/phase/<int:phase_number>')
def show_phase(phase_number):
    """Mostrar detalles de una fase específica"""
    phase_files = {
        1: 'phase1_data_understanding.json',
        2: 'phase2_exploration.json',
        3: 'phase3_preparation.json',
        4: 'phase4_modeling.json',
        5: 'phase5_evaluation.json'
    }
    
    if phase_number not in phase_files:
        return jsonify({'success': False, 'error': 'Fase no válida'}), 404
    
    phase_data = load_precomputed_data(phase_files[phase_number])
    
    if not phase_data:
        return jsonify({
            'success': False, 
            'error': f'Datos de fase {phase_number} no disponibles'
        }), 404
    
    return jsonify({
        'success': True,
        'phase': phase_number,
        'data': phase_data
    })


@bp.route('/api/phase1')
def api_phase1():
    """API: Fase 1 - Comprensión de Datos"""
    data = load_precomputed_data('phase1_data_understanding.json')
    
    if not data:
        return jsonify({'success': False, 'error': 'Datos no disponibles'}), 404
    
    return jsonify({
        'success': True,
        'phase': 'Comprensión de Datos',
        'data': data
    })


@bp.route('/api/phase2')
def api_phase2():
    """API: Fase 2 - Exploración de Datos"""
    data = load_precomputed_data('phase2_exploration.json')
    visualizations = load_visualizations()
    
    if not data:
        return jsonify({'success': False, 'error': 'Datos no disponibles'}), 404
    
    # Convertir visualizaciones a JSON
    viz_json = {}
    if visualizations and 'exploration' in visualizations:
        for key, fig in visualizations['exploration']['visualizations'].items():
            if fig is not None and hasattr(fig, 'to_json'):
                viz_json[key] = fig.to_json()
    
    return jsonify({
        'success': True,
        'phase': 'Exploración de Datos',
        'data': data,
        'visualizations': viz_json
    })


@bp.route('/api/phase3')
def api_phase3():
    """API: Fase 3 - Preparación de Datos"""
    data = load_precomputed_data('phase3_preparation.json')
    
    if not data:
        return jsonify({'success': False, 'error': 'Datos no disponibles'}), 404
    
    return jsonify({
        'success': True,
        'phase': 'Preparación de Datos',
        'data': data
    })


@bp.route('/api/phase4')
def api_phase4():
    """API: Fase 4 - Modelado"""
    data = load_precomputed_data('phase4_modeling.json')
    
    if not data:
        return jsonify({'success': False, 'error': 'Datos no disponibles'}), 404
    
    return jsonify({
        'success': True,
        'phase': 'Modelado',
        'data': data
    })


@bp.route('/api/phase5')
def api_phase5():
    """API: Fase 5 - Evaluación"""
    data = load_precomputed_data('phase5_evaluation.json')
    visualizations = load_visualizations()
    
    if not data:
        return jsonify({'success': False, 'error': 'Datos no disponibles'}), 404
    
    # Convertir visualizaciones a JSON
    viz_json = {}
    if visualizations:
        for key, fig in visualizations.items():
            if key == 'comparison_table' or key == 'best_model_name':
                continue
            if fig is not None and hasattr(fig, 'to_json'):
                viz_json[key] = fig.to_json()
    
    return jsonify({
        'success': True,
        'phase': 'Evaluación',
        'data': data,
        'visualizations': viz_json
    })


@bp.route('/results')
def results():
    """Página de resultados completos"""
    evaluation_data = load_precomputed_data('phase5_evaluation.json')
    
    if not evaluation_data:
        return render_template('results.html', 
                             data_available=False,
                             error_message="Datos no disponibles")
    
    return render_template('results.html',
                         data_available=True,
                         evaluation=evaluation_data)


@bp.route('/about')
def about():
    """Página sobre CRISP-DM"""
    return render_template('about.html')


@bp.route('/api/status')
def status():
    """Estado de los datos precalculados"""
    phases = {
        'phase1': RESULTS_DIR / 'phase1_data_understanding.json',
        'phase2': RESULTS_DIR / 'phase2_exploration.json',
        'phase3': RESULTS_DIR / 'phase3_preparation.json',
        'phase4': RESULTS_DIR / 'phase4_modeling.json',
        'phase5': RESULTS_DIR / 'phase5_evaluation.json',
        'visualizations': RESULTS_DIR / 'visualizations.pkl',
        'best_model': RESULTS_DIR / 'best_model.pkl'
    }
    
    status_info = {
        phase: path.exists() 
        for phase, path in phases.items()
    }
    
    status_info['all_ready'] = all(status_info.values())
    
    return jsonify(status_info)