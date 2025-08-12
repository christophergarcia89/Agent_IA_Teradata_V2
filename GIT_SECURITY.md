# 🔐 GUÍA DE SEGURIDAD GIT - Agent IA Teradata

## ⚠️ Archivos que NUNCA deben subirse a GitHub

### 📁 Directorios completos que se ignoran:
- `venv/` - Entorno virtual de Python
- `logs/` - Archivos de log del sistema  
- `data/chroma_db/` - Base de datos ChromaDB
- `__pycache__/` - Cache de Python

### 🔑 Archivos de configuración sensibles:
- `.env` - Variables de entorno
- `config/secrets.py` - Configuraciones secretas
- `*.key`, `*.pem`, `*.crt` - Certificados y claves
- `*.log` - Archivos de log individuales

## 🛡️ Validación antes de commit

### Método 1: Script automático (Recomendado)
```bash
# Ejecutar antes de cada commit
.\validate_git.bat
```

### Método 2: Validación manual
```bash
# 1. Verificar que archivos sensibles NO están rastreados
git ls-files | findstr /i "venv env .env secrets logs"

# 2. Verificar el estado actual
git status

# 3. Ver qué archivos se van a subir
git diff --cached --name-only
```

## ✅ Proceso de commit seguro

1. **Validar seguridad:**
   ```bash
   .\validate_git.bat
   ```

2. **Revisar cambios:**
   ```bash
   git status
   git diff
   ```

3. **Hacer commit:**
   ```bash
   git add .
   git commit -m "Descripción del cambio"
   git push origin main
   ```

## 🚨 Si accidentalmente subes archivos sensibles:

### Para archivos aún no commiteados:
```bash
# Quitar del staging area
git reset HEAD <archivo_sensible>

# O quitar todos
git reset HEAD .
```

### Para archivos ya commiteados pero no pusheados:
```bash
# Eliminar del último commit
git rm --cached <archivo_sensible>
git commit --amend -m "Remove sensitive files"
```

### Para archivos ya pusheados a GitHub:
```bash
# 1. Eliminar del repositorio (mantener local)
git rm --cached <archivo_sensible>
git commit -m "Remove sensitive files from tracking"
git push origin main

# 2. Para eliminar del historial completamente (PELIGROSO):
git filter-branch --force --index-filter \
'git rm --cached --ignore-unmatch <archivo_sensible>' \
--prune-empty --tag-name-filter cat -- --all

git push origin --force --all
```

## 🔍 Comandos útiles de verificación

```bash
# Ver todos los archivos rastreados
git ls-files

# Ver archivos ignorados  
git ls-files --ignored --exclude-standard

# Verificar que .gitignore funciona
echo "test_secret=123" > .env_test
git status  # No debería aparecer .env_test
rm .env_test

# Ver historial de un archivo
git log --follow <archivo>
```

## ⚙️ Configuración del .gitignore

El archivo `.gitignore` ya está configurado correctamente con:

- ✅ Entornos virtuales Python
- ✅ Variables de entorno (.env*)
- ✅ Archivos de configuración sensibles
- ✅ Logs y bases de datos
- ✅ Cache y archivos temporales
- ✅ Archivos específicos del IDE

## 🎯 Mejores prácticas

1. **Siempre ejecuta `.\validate_git.bat` antes de commit**
2. **Revisa `git status` antes de hacer push**
3. **Nunca hardcodees credenciales en el código**
4. **Usa variables de entorno para configuración sensible**
5. **Mantén el .gitignore actualizado**
6. **Haz commits pequeños y frecuentes**

## 📞 En caso de emergencia

Si accidentalmente subiste información sensible:

1. **¡NO ENTRES EN PÁNICO!**
2. Ejecuta inmediatamente los comandos de limpieza de arriba
3. Cambia todas las credenciales/secrets expuestos
4. Considera hacer el repositorio privado temporalmente
5. Documenta el incidente para prevenir futuras ocurrencias

---
*Última actualización: $(date)*
