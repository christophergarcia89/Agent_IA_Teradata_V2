#!/usr/bin/env python3
"""
Script de validación para asegurar que archivos sensibles no se suban a Git
Ejecutar antes de hacer commit para verificar seguridad del repositorio
"""

import os
import subprocess
import sys
from pathlib import Path

# Lista de archivos/directorios que NUNCA deben subirse
SENSITIVE_FILES = [
    'venv/',
    'env/',
    '.venv/',
    '.env',
    '.env.local',
    '.env.development.local', 
    '.env.test.local',
    '.env.production.local',
    'config/secrets.py',
    'config/.secrets',
    'logs/',
    '*.log',
    'data/chroma_db/',
    '*.db',
    '*.sqlite3',
    '*.key',
    '*.pem',
    '*.crt',
    '__pycache__/',
    '*.pyc',
    '.cache/',
    'node_modules/'
]

def check_git_repo():
    """Verificar que estamos en un repositorio Git."""
    if not Path('.git').exists():
        print("❌ Este directorio no es un repositorio Git")
        return False
    return True

def check_gitignore_exists():
    """Verificar que .gitignore existe."""
    gitignore_path = Path('.gitignore')
    if not gitignore_path.exists():
        print("❌ Archivo .gitignore no existe")
        return False
    print("✅ Archivo .gitignore existe")
    return True

def check_gitignore_content():
    """Verificar que .gitignore contiene las reglas necesarias."""
    gitignore_path = Path('.gitignore')
    
    try:
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            content = f.read().lower()
        
        missing_rules = []
        for sensitive_file in SENSITIVE_FILES:
            # Convertir patrones para comparación
            search_pattern = sensitive_file.lower().replace('*', '').replace('/', '')
            if search_pattern not in content:
                missing_rules.append(sensitive_file)
        
        if missing_rules:
            print("⚠️  Reglas faltantes en .gitignore:")
            for rule in missing_rules[:5]:  # Solo mostrar primeras 5
                print(f"   - {rule}")
            return False
        
        print("✅ .gitignore contiene las reglas necesarias")
        return True
        
    except Exception as e:
        print(f"❌ Error leyendo .gitignore: {e}")
        return False

def get_tracked_files():
    """Obtener lista de archivos rastreados por Git."""
    try:
        result = subprocess.run(['git', 'ls-files'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip().split('\n') if result.stdout.strip() else []
    except subprocess.CalledProcessError:
        print("❌ Error ejecutando 'git ls-files'")
        return None

def check_sensitive_files_tracked():
    """Verificar que ningún archivo sensible está siendo rastreado."""
    tracked_files = get_tracked_files()
    if tracked_files is None:
        return False
    
    # Convertir patrones de archivos sensibles para comparación
    sensitive_patterns = []
    for pattern in SENSITIVE_FILES:
        clean_pattern = pattern.replace('*', '').replace('/', '').lower()
        sensitive_patterns.append(clean_pattern)
    
    tracked_sensitive = []
    for file in tracked_files:
        file_lower = file.lower()
        for pattern in sensitive_patterns:
            if pattern in file_lower:
                tracked_sensitive.append(file)
                break
    
    if tracked_sensitive:
        print("❌ ARCHIVOS SENSIBLES SIENDO RASTREADOS:")
        for file in tracked_sensitive[:10]:  # Máximo 10
            print(f"   - {file}")
        print("\n🔧 Para eliminarlos del seguimiento:")
        print("   git rm --cached <archivo>")
        print("   git commit -m 'Remove sensitive files from tracking'")
        return False
    
    print("✅ Ningún archivo sensible está siendo rastreado")
    return True

def check_untracked_sensitive():
    """Verificar archivos sensibles no rastreados que podrían subirse por error."""
    try:
        # Obtener archivos sin rastrar
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True, check=True)
        untracked_lines = [line for line in result.stdout.split('\n') 
                          if line.startswith('??')]
        
        untracked_sensitive = []
        for line in untracked_lines:
            file = line[3:].strip()  # Quitar '?? '
            file_lower = file.lower()
            
            for pattern in SENSITIVE_FILES:
                clean_pattern = pattern.replace('*', '').replace('/', '').lower()
                if clean_pattern in file_lower:
                    untracked_sensitive.append(file)
                    break
        
        if untracked_sensitive:
            print("⚠️  Archivos sensibles sin rastrar encontrados:")
            for file in untracked_sensitive[:5]:
                print(f"   - {file}")
            print("   (Deberían estar siendo ignorados por .gitignore)")
            return False
        
        return True
        
    except subprocess.CalledProcessError:
        print("❌ Error ejecutando 'git status'")
        return False

def main():
    """Función principal de validación."""
    print("🔍 VALIDACIÓN DE SEGURIDAD GIT")
    print("=" * 35)
    print()
    
    # Lista de verificaciones
    checks = [
        ("Repositorio Git", check_git_repo),
        ("Archivo .gitignore", check_gitignore_exists),
        ("Contenido .gitignore", check_gitignore_content),
        ("Archivos sensibles rastreados", check_sensitive_files_tracked),
        ("Archivos sensibles sin rastrar", check_untracked_sensitive),
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"🔍 Verificando: {check_name}")
        if not check_func():
            all_passed = False
        print()
    
    # Resultado final
    print("=" * 35)
    if all_passed:
        print("🎉 ¡VALIDACIÓN EXITOSA!")
        print("   Es seguro hacer commit y push")
        sys.exit(0)
    else:
        print("💥 VALIDACIÓN FALLÓ")
        print("   Revisa los errores antes de subir el código")
        sys.exit(1)

if __name__ == "__main__":
    main()
