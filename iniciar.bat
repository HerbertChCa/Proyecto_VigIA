@echo off
REM VigIA - inicio en Windows. Uso: doble clic, o "iniciar.bat --demo" para la escena simulada.
chcp 65001 >nul
cd /d "%~dp0"

set PY=python
where py >nul 2>nul && set PY=py -3

if not exist ".venv\Scripts\python.exe" (
  echo Creando el entorno virtual de Python...
  %PY% -m venv .venv
  if errorlevel 1 goto error
)
call ".venv\Scripts\activate.bat"

if not exist ".venv\instalado.txt" (
  echo Instalando dependencias. La primera vez tarda varios minutos...
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  if errorlevel 1 goto error
  echo ok > ".venv\instalado.txt"
)

if /I not "%~1"=="--demo" (
  if not exist "models\yolo11n-pose.pt" (
    echo Descargando modelos...
    python scripts\descargar_modelos.py
  )
)

echo.
echo Abriendo VigIA en http://127.0.0.1:8000  -  para detenerlo, cierre esta ventana.
python -m vigia %*
goto fin

:error
echo.
echo Ocurrio un error. Revise el mensaje de arriba.
pause
:fin
