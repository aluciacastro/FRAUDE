# -*- coding: utf-8 -*-
"""
Archivo principal para ejecutar la aplicación Flask de Detección de Fraude
"""
from flask import Flask
from config.settings import SECRET_KEY, DEBUG, HOST, PORT
import os


def create_app():
    """Factory para crear la aplicación Flask"""
    # Especificar la carpeta app como base
    app = Flask(
        __name__,
        template_folder='app/templates',
        static_folder='app/static'
    )

    # Configuración
    app.config['SECRET_KEY'] = SECRET_KEY
    app.config['DEBUG'] = DEBUG
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

    # Registrar filtros personalizados de Jinja2
    @app.template_filter('number_format')
    def number_format(value):
        """Formatear números con separadores de miles"""
        try:
            return "{:,}".format(int(value))
        except (ValueError, TypeError):
            return value

    # Registrar blueprints
    from app.routes.main_routes import bp as main_bp
    app.register_blueprint(main_bp)

    # Crear directorios necesarios
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/precomputed', exist_ok=True)
    os.makedirs('models_saved', exist_ok=True)

    return app


if __name__ == '__main__':
    app = create_app()

    print("="*60)
    print(" Sistema de Detección de Fraude")
    print("="*60)
    print(f" Servidor corriendo en: http://{HOST}:{PORT}")
    print(f" Modo Debug: {DEBUG}")
    print("="*60)
    print("\nPresiona CTRL+C para detener el servidor\n")

    app.run(host=HOST, port=PORT, debug=DEBUG)