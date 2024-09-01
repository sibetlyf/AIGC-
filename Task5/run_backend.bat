@echo off
setlocal

:: Python路径
set "python_path=%cd%\py310"

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
set FFMPEG_PATH=%cd%\py310\ffmpeg\bin
set HF_ENDPOINT=https://hf-mirror.com
set HF_HOME=%cd%\hf_download

:: 修改系统路径
set PATH=%python_path%;%python_path%\Scripts;%FFMPEG_PATH%;%PATH%

:: 执行Python脚本
%PYTHON_EXECUTABLE% stream\backend.py

endlocal
