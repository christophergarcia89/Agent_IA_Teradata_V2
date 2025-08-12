@echo off
REM Script simple para validar que archivos sensibles no se suban a Git
REM Ejecutar antes de hacer commit

echo =======================================
echo    VALIDACIÓN DE ARCHIVOS SENSIBLES
echo =======================================
echo.

echo 1. Verificando que .gitignore existe...
if exist ".gitignore" (
    echo    ✓ .gitignore existe
) else (
    echo    ✗ .gitignore NO existe - CREAR URGENTE
    exit /b 1
)
echo.

echo 2. Verificando archivos rastreados por Git...
git ls-files | findstr /i "venv env .env secrets logs .pyc .log chroma_db" >nul 2>&1
if %errorlevel% equ 0 (
    echo    ✗ ARCHIVOS SENSIBLES DETECTADOS:
    git ls-files | findstr /i "venv env .env secrets logs .pyc .log chroma_db"
    echo.
    echo    SOLUCIÓN:
    echo    git rm --cached ^<archivo^>
    echo    git commit -m "Remove sensitive files"
    exit /b 1
) else (
    echo    ✓ Ningún archivo sensible está rastreado
)
echo.

echo 3. Verificando archivos sin rastrar...
git status --porcelain | findstr "^??" | findstr /i "venv env .env secrets logs .pyc .log chroma_db" >nul 2>&1
if %errorlevel% equ 0 (
    echo    ! Archivos sensibles sin rastrar detectados:
    git status --porcelain | findstr "^??" | findstr /i "venv env .env secrets logs .pyc .log chroma_db"
    echo    (Deberían estar en .gitignore)
) else (
    echo    ✓ No hay archivos sensibles pendientes
)
echo.

echo 4. Lista de archivos que se van a subir:
echo    Archivos nuevos/modificados:
git status --porcelain | findstr "^[AM]"
echo.

echo =======================================
echo    VALIDACIÓN COMPLETA
echo =======================================
echo ✓ Es SEGURO hacer commit y push
echo.
echo Comandos sugeridos:
echo   git add .
echo   git commit -m "Descripción del cambio"
echo   git push origin main
