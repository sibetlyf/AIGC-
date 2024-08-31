#!/bin/bash

# 当前目录
current_dir=$(pwd)

# Python路径
python_path="$current_dir/py310"

# 设置环境变量
export PYTHON_PATH="$python_path"
export PYTHONHOME=""
export PYTHONPATH=""
export PYTHONEXECUTABLE="$python_path/python"
export PYTHONWEXECUTABLE="$python_path/pythonw"
export PYTHON_EXECUTABLE="$python_path/python"
export PYTHONW_EXECUTABLE="$python_path/pythonw"
export PYTHON_BIN_PATH="$python_path/python"
export PYTHON_LIB_PATH="$python_path/Lib/site-packages"
export FFMPEG_PATH="$current_dir/py311/ffmpeg/bin"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="$current_dir/hf_download"

# 修改系统路径
export PATH="$python_path:$python_path/Scripts:$FFMPEG_PATH:$PATH"

# 执行Python脚本
nohup "$PYTHON_EXECUTABLE" stream/app.py &
nohup "$PYTHON_EXECUTABLE" stream/backend.py &

echo "成功启动"