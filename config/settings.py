import os
from pathlib import Path

# Directorios base
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models_saved'
STATIC_DIR = BASE_DIR / 'app' / 'static'
TEMPLATES_DIR = BASE_DIR / 'app' / 'templates'

# Crear directorios si no existen
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Configuración de Flask
SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
DEBUG = os.environ.get('DEBUG', 'True') == 'True'
HOST = os.environ.get('HOST', '0.0.0.0')
PORT = int(os.environ.get('PORT', 5000))

# Configuración de datos
DATASET_PATH = DATA_DIR / 'creditcard.csv'
MAX_UPLOAD_SIZE = 16 * 1024 * 1024  # 16MB

# Configuración de modelos
MODELS_CONFIG = {
    'Logistic Regression': {
        'class': 'LogisticRegression',
        'params': {'random_state': 42, 'max_iter': 1000}
    },
    'Decision Tree': {
        'class': 'DecisionTreeClassifier',
        'params': {'random_state': 42, 'max_depth': 10}
    },
    'Random Forest': {
        'class': 'RandomForestClassifier',
        'params': {'random_state': 42, 'n_estimators': 100, 'max_depth': 15}
    },
    'Gradient Boosting': {
        'class': 'GradientBoostingClassifier',
        'params': {'random_state': 42, 'n_estimators': 100, 'max_depth': 5}
    }
}

# Estrategias de balanceo
BALANCING_STRATEGIES = {
    'Original': {'type': 'none'},
    'SMOTE': {'type': 'smote', 'sampling_strategy': 0.5},
    'Under-sampling': {'type': 'rus', 'sampling_strategy': 0.5}
}

# Configuración de entrenamiento
TRAIN_TEST_SPLIT = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Configuración de visualización
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
PLOT_DPI = 100
FIGURE_SIZE = (12, 8)

# Colores para gráficos
COLORS = {
    'normal': '#2ecc71',
    'fraud': '#e74c3c',
    'primary': '#3498db',
    'secondary': '#95a5a6'
}
