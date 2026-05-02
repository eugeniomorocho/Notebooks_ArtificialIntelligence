#!/usr/bin/env python3
"""
Script para diagnosticar problemas del entorno Python
"""

import sys
import subprocess
from importlib import util

def check_package(package_name):
    """Verifica si un paquete está instalado"""
    spec = util.find_spec(package_name)
    return spec is not None

def get_package_version(package_name):
    """Obtiene la versión de un paquete"""
    try:
        module = __import__(package_name)
        return getattr(module, '__version__', 'Unknown')
    except ImportError:
        return 'Not installed'

def main():
    print("🔍 DIAGNÓSTICO DEL ENTORNO PYTHON")
    print("=" * 50)

    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")

    # Paquetes críticos a verificar
    critical_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn',
        'sklearn', 'optuna', 'torch', 'PIL',
        'xgboost', 'lightgbm', 'catboost', 'shap'
    ]

    print("\n📦 ESTADO DE PAQUETES:")
    print("-" * 30)

    for package in critical_packages:
        installed = check_package(package)
        if installed:
            version = get_package_version(package)
            print(f"✅ {package:<15} {version}")
        else:
            print(f"❌ {package:<15} NOT INSTALLED")

    print("\n🔧 COMANDOS DE REPARACIÓN:")
    print("-" * 30)
    print("Si hay paquetes faltantes, ejecuta:")
    print("pip install numpy pandas matplotlib seaborn scikit-learn")
    print("pip install optuna torch torchvision pillow")
    print("pip install xgboost lightgbm catboost shap")

    print("\n💡 SOLUCIÓN RÁPIDA:")
    print("1. Restart kernel: Kernel → Restart Kernel")
    print("2. Reinstall packages: !pip install --upgrade [package_name]")
    print("3. Verify virtual env: Check if using correct Python environment")

if __name__ == "__main__":
    main()