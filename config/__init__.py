# -*- coding: utf-8 -*-
"""
Módulo de configuración
"""
from .settings import *

class Config:
    """Clase de configuración para Flask"""
    SECRET_KEY = SECRET_KEY
    DEBUG = DEBUG
    HOST = HOST
    PORT = PORT
    DATASET_PATH = DATASET_PATH
    DATA_DIR = DATA_DIR
    MODELS_DIR = MODELS_DIR