// JavaScript principal para la aplicación de Detección de Fraude

// Configuración global
const CONFIG = {
    API_BASE_URL: '',
    CHART_THEME: 'plotly_white'
};

// Utilidades
const Utils = {
    // Mostrar notificación
    showNotification(message, type = 'info') {
        const alertClass = `alert-${type}`;
        const alert = document.createElement('div');
        alert.className = `alert ${alertClass} alert-dismissible fade show position-fixed top-0 end-0 m-3`;
        alert.style.zIndex = '9999';
        alert.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        document.body.appendChild(alert);

        setTimeout(() => {
            alert.remove();
        }, 5000);
    },

    // Formatear número
    formatNumber(num, decimals = 2) {
        return Number(num).toFixed(decimals);
    },

    // Formatear número con separador de miles
    formatNumberWithCommas(num) {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    },

    // Mostrar/ocultar loading
    toggleLoading(elementId, show = true) {
        const element = document.getElementById(elementId);
        if (element) {
            element.style.display = show ? 'block' : 'none';
        }
    },

    // Scroll suave a elemento
    scrollToElement(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }
};

// Manejo de gráficos Plotly
const ChartManager = {
    // Renderizar gráfico Plotly
    renderPlotly(divId, plotData) {
        try {
            if (!plotData) {
                console.error(`No hay datos para renderizar en ${divId}`);
                return;
            }

            const data = JSON.parse(plotData);
            Plotly.newPlot(divId, data.data, data.layout, {
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToRemove: ['toImage'],
                displaylogo: false
            });
        } catch (error) {
            console.error(`Error renderizando gráfico en ${divId}:`, error);
        }
    },

    // Limpiar gráfico
    clearPlot(divId) {
        Plotly.purge(divId);
    },

    // Actualizar gráfico
    updatePlot(divId, plotData) {
        this.clearPlot(divId);
        this.renderPlotly(divId, plotData);
    }
};

// Manejo de la API
const API = {
    // Realizar petición GET
    async get(endpoint) {
        try {
            const response = await fetch(`${CONFIG.API_BASE_URL}${endpoint}`);
            return await this.handleResponse(response);
        } catch (error) {
            console.error('Error en GET:', error);
            throw error;
        }
    },

    // Realizar petición POST
    async post(endpoint, data) {
        try {
            const response = await fetch(`${CONFIG.API_BASE_URL}${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
            return await this.handleResponse(response);
        } catch (error) {
            console.error('Error en POST:', error);
            throw error;
        }
    },

    // Subir archivo
    async uploadFile(endpoint, formData) {
        try {
            const response = await fetch(`${CONFIG.API_BASE_URL}${endpoint}`, {
                method: 'POST',
                body: formData
            });
            return await this.handleResponse(response);
        } catch (error) {
            console.error('Error en upload:', error);
            throw error;
        }
    },

    // Manejar respuesta
    async handleResponse(response) {
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Error en la petición');
        }
        return await response.json();
    }
};

// Validación de formularios
const FormValidator = {
    // Validar archivo CSV
    validateCSVFile(file) {
        if (!file) {
            return { valid: false, message: 'Por favor seleccione un archivo' };
        }

        if (!file.name.endsWith('.csv')) {
            return { valid: false, message: 'El archivo debe ser un CSV' };
        }

        const maxSize = 100 * 1024 * 1024; // 100MB
        if (file.size > maxSize) {
            return { valid: false, message: 'El archivo es demasiado grande (máx 100MB)' };
        }

        return { valid: true };
    },

    // Validar campo requerido
    validateRequired(value, fieldName) {
        if (!value || value.trim() === '') {
            return { valid: false, message: `${fieldName} es requerido` };
        }
        return { valid: true };
    }
};

// Manejo del estado de la aplicación
const AppState = {
    dataLoaded: false,
    dataAnalyzed: false,
    modelsTrained: false,

    // Actualizar estado
    setState(updates) {
        Object.assign(this, updates);
        this.updateUI();
    },

    // Actualizar UI según estado
    updateUI() {
        // Habilitar/deshabilitar botones según el estado
        const analyzeBtn = document.getElementById('analyzeBtn');
        if (analyzeBtn) {
            analyzeBtn.disabled = !this.dataLoaded;
        }
    },

    // Obtener estado del servidor
    async fetchServerState() {
        try {
            const state = await API.get('/api/status');
            this.setState(state);
        } catch (error) {
            console.error('Error obteniendo estado:', error);
        }
    }
};

// Manejo de resultados
const ResultsManager = {
    // Renderizar tabla de comparación
    renderComparisonTable(data) {
        if (!data || !Array.isArray(data)) return '';

        let html = `
            <div class="table-responsive">
                <table class="table table-striped table-hover">
                    <thead>
                        <tr>
                            <th>Modelo</th>
                            <th>Accuracy</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F1-Score</th>
                            <th>ROC-AUC</th>
                        </tr>
                    </thead>
                    <tbody>
        `;

        data.forEach(row => {
            html += `
                <tr>
                    <td><strong>${row.Model}</strong></td>
                    <td>${Utils.formatNumber(row.Accuracy, 4)}</td>
                    <td>${Utils.formatNumber(row.Precision, 4)}</td>
                    <td>${Utils.formatNumber(row.Recall, 4)}</td>
                    <td>${Utils.formatNumber(row['F1-Score'], 4)}</td>
                    <td>${Utils.formatNumber(row['ROC-AUC'], 4)}</td>
                </tr>
            `;
        });

        html += '</tbody></table></div>';
        return html;
    },

    // Renderizar métricas del mejor modelo
    renderBestModelMetrics(modelName, metrics) {
        return `
            <div class="alert alert-success">
                <h5><i class="fas fa-trophy"></i> Mejor Modelo: ${modelName}</h5>
                <div class="row mt-3">
                    <div class="col-md-4">
                        <div class="metric-card">
                            <div class="metric-value">${Utils.formatNumber(metrics.f1_score, 3)}</div>
                            <div class="metric-label">F1-Score</div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="metric-card">
                            <div class="metric-value">${Utils.formatNumber(metrics.recall, 3)}</div>
                            <div class="metric-label">Recall</div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="metric-card">
                            <div class="metric-value">${Utils.formatNumber(metrics.roc_auc, 3)}</div>
                            <div class="metric-label">ROC-AUC</div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
};

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    console.log('Aplicación de Detección de Fraude iniciada');

    // Obtener estado inicial
    AppState.fetchServerState();

    // Tooltips de Bootstrap
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});

// Exportar funciones globales
window.Utils = Utils;
window.ChartManager = ChartManager;
window.API = API;
window.FormValidator = FormValidator;
window.AppState = AppState;
window.ResultsManager = ResultsManager;
