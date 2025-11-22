# 🛡️ Sistema de Detección de Fraude con Machine Learning

Aplicación web completa para detectar transacciones fraudulentas usando Machine Learning y la metodología CRISP-DM.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📊 Características

- ✅ Análisis exploratorio de datos interactivo
- ✅ Preprocesamiento con técnicas de balanceo (SMOTE, Under-sampling)
- ✅ Entrenamiento de múltiples modelos (Logistic Regression, Random Forest, XGBoost, Gradient Boosting)
- ✅ Evaluación con métricas especializadas (Precision, Recall, F1-Score, ROC-AUC)
- ✅ Visualizaciones interactivas con Plotly
- ✅ Navegación por fases CRISP-DM
- ✅ Arquitectura modular con principios SOLID

## 🚀 Instalación Rápida

### 1. Clonar el repositorio
```bash
git clone https://github.com/aluciacastro/FRAUDE.git
cd FRAUDE
```

### 2. Crear entorno virtual
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Descargar el dataset
Descarga el dataset **Credit Card Fraud Detection** desde Kaggle:
- 🔗 [https://www.kaggle.com/mlg-ulb/creditcardfraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Guarda el archivo `creditcard.csv` en la carpeta `data/`

### 5. Generar datos precalculados
```bash
python notebook_processor.py
```
Este script:
- Carga el dataset
- Ejecuta análisis exploratorio
- Entrena todos los modelos
- Genera visualizaciones
- Guarda resultados en `data/precomputed/`

⏱️ **Tiempo estimado:** 5-10 minutos (dependiendo de tu hardware)

### 6. Ejecutar la aplicación
```bash
python run.py
```

Abre tu navegador en: **http://localhost:5000**

## 📁 Estructura del Proyecto

```
FRAUDE/
├── app/
│   ├── models/              # Modelos de ML
│   │   ├── ml_models.py     # Logistic Regression, Random Forest, XGBoost, etc.
│   │   └── __init__.py
│   ├── pipelines/           # Pipeline completo CRISP-DM
│   │   ├── fraud_pipeline.py
│   │   └── __init__.py
│   ├── routes/              # Rutas Flask
│   │   ├── main_routes.py
│   │   └── __init__.py
│   ├── services/            # Servicios de datos y visualización
│   │   ├── data_loader.py
│   │   ├── data_preprocessor.py
│   │   ├── visualizer.py
│   │   └── __init__.py
│   ├── static/              # CSS y JS
│   │   └── css/
│   │       └── style.css
│   └── templates/           # Templates HTML
│       ├── base.html
│       ├── index.html
│       └── results.html
├── config/                  # Configuración
│   ├── settings.py
│   └── __init__.py
├── data/                    # Datasets
│   ├── creditcard.csv       # (descargar de Kaggle)
│   └── precomputed/         # Datos precalculados
├── notebook/                # Jupyter Notebook
│   └── deteccion_fraude_creditcard.ipynb
├── notebook_processor.py    # Script para generar datos
├── requirements.txt
├── run.py                   # Punto de entrada
└── README.md
```

## 🎯 Metodología CRISP-DM

El proyecto sigue las 6 fases de CRISP-DM:

### 1️⃣ Comprensión del Negocio
- Problema: Detectar fraudes en tarjetas de crédito
- Objetivo: Minimizar fraudes no detectados y falsos positivos

### 2️⃣ Comprensión de Datos
- Dataset: 284,807 transacciones
- Features: 30 (V1-V28 transformadas con PCA, Time, Amount)
- Target: Class (0=Normal, 1=Fraude)
- Desbalance: 0.172% fraudes

### 3️⃣ Preparación de Datos
- Limpieza de duplicados y nulos
- Escalado con RobustScaler
- Balanceo con SMOTE
- División 80/20 (train/test)

### 4️⃣ Modelado
Modelos entrenados:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

### 5️⃣ Evaluación
Métricas principales:
- **Recall**: Capacidad de detectar fraudes reales
- **Precision**: Evitar falsos positivos
- **F1-Score**: Balance entre precision y recall
- **ROC-AUC**: Capacidad general de discriminación

### 6️⃣ Despliegue
- Aplicación web Flask
- Visualizaciones interactivas
- Navegación por fases

## 📊 Tecnologías Utilizadas

### Backend
- **Flask 3.0.0**: Framework web
- **scikit-learn**: Modelos de ML
- **XGBoost**: Gradient Boosting optimizado
- **imbalanced-learn**: Técnicas de balanceo (SMOTE)

### Visualización
- **Plotly**: Gráficos interactivos
- **Matplotlib & Seaborn**: Gráficos estáticos (notebook)

### Frontend
- **Bootstrap 5**: Framework CSS
- **Font Awesome**: Iconos
- **JavaScript**: Interactividad

## 🔧 Configuración

Edita `config/settings.py` para personalizar:

```python
# Rutas
DATASET_PATH = DATA_DIR / 'creditcard.csv'

# Parámetros de entrenamiento
TRAIN_TEST_SPLIT = 0.2
RANDOM_STATE = 42

# Modelos
MODELS_CONFIG = {
    'Random Forest': {
        'n_estimators': 100,
        'max_depth': 15
    }
}
```

## 📈 Resultados Esperados

Con el dataset de ejemplo, los mejores modelos alcanzan:
- **Accuracy**: ~99.9%
- **Precision**: ~85-90%
- **Recall**: ~75-85%
- **F1-Score**: ~80-87%
- **ROC-AUC**: ~95-98%

⚠️ **Nota**: El recall es la métrica más importante para fraudes (detectar todos los casos reales).

## 🐛 Solución de Problemas

### Error: "Dataset no encontrado"
```bash
# Verifica que creditcard.csv esté en data/
ls data/creditcard.csv

# Si no existe, descárgalo de Kaggle
```

### Error: "Datos precalculados no disponibles"
```bash
# Ejecuta el generador de datos
python notebook_processor.py
```

### Error: Módulo no encontrado
```bash
# Reinstala las dependencias
pip install -r requirements.txt --force-reinstall
```

### Puerto 5000 en uso
```python
# Edita run.py y cambia el puerto
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
```

## 📝 Uso de la Aplicación

1. **Inicio**: Visualiza las fases CRISP-DM y estadísticas del dataset
2. **Haz clic en cada fase**: Abre un modal con información detallada
3. **Explora visualizaciones**: Gráficos interactivos con Plotly
4. **Compara modelos**: Tabla comparativa de métricas
5. **Analiza resultados**: Curvas ROC, matrices de confusión

## 🤝 Contribuciones

Las contribuciones son bienvenidas:

1. Fork el proyecto
2. Crea una rama (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👤 Autor

**Alucia Castro**
- GitHub: [@aluciacastro](https://github.com/aluciacastro)

## 🙏 Agradecimientos

- Dataset: [Machine Learning Group - ULB](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Metodología: CRISP-DM
- Comunidad: scikit-learn, Flask, Plotly

## 📚 Referencias

- [CRISP-DM Methodology](https://www.datascience-pm.com/crisp-dm-2/)
- [Handling Imbalanced Datasets](https://imbalanced-learn.org/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)

---

⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub!