@echo off
setlocal

:: 当前目录
set "current_dir=%cd%"

:: Python路径
set "python_path=%current_dir%\py310"

:: 设置环境变量
set PYTHON_PATH=%python_path%
set PYTHONHOME=
set PYTHONPATH=
set PYTHONEXECUTABLE=%python_path%\python.exe
set PYTHONWEXECUTABLE=%python_path%pythonw.exe
set PYTHON_EXECUTABLE=%python_path%\python.exe
set PYTHONW_EXECUTABLE=%python_path%pythonw.exe
set PYTHON_BIN_PATH=%python_path%\python.exe
set PYTHON_LIB_PATH=%python_path%\Lib\site-packages
set FFMPEG_PATH=%current_dir%\py310\ffmpeg\bin
set HF_ENDPOINT=https://hf-mirror.com
set HF_HOME=%current_dir%\hf_download

:: 修改系统路径
set PATH=%python_path%;%python_path%\Scripts;%FFMPEG_PATH%;%PATH%

:: 启动 run_app.bat 和 run_backend.bat
(
    start "" cmd /k "call run_backend.bat"
    start "" cmd /k "call run_app.bat"
)

echo 成功启动 main.bat
endlocal
pause