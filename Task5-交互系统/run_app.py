import os
import subprocess

# 当前目录
current_dir = os.getcwd()

# Python路径
python_path = os.path.join(current_dir, 'py310')

# 设置环境变量
os.environ['PYTHON_PATH'] = f'{python_path}\\'
os.environ['PYTHONHOME'] = ''
os.environ['PYTHONPATH'] = ''
os.environ['PYTHONEXECUTABLE'] = f'{python_path}\\python.exe'
os.environ['PYTHONWEXECUTABLE'] = f'{python_path}pythonw.exe'
os.environ['PYTHON_EXECUTABLE'] = f'{python_path}\\python.exe'
os.environ['PYTHONW_EXECUTABLE'] = f'{python_path}pythonw.exe'
os.environ['PYTHON_BIN_PATH'] = f'{python_path}\\python.exe'
os.environ['PYTHON_LIB_PATH'] = f'{python_path}\\Lib\\site-packages'
os.environ['FFMPEG_PATH'] = f'{current_dir}\\py311\\ffmpeg\\bin'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = f'{current_dir}\\hf_download'
# os.environ['PYTHONPATH'] = 'tts/third_party/AcademiCodec;third_party/Matcha-TTS'

# 修改系统路径
os.environ['PATH'] = f'{python_path};{python_path}\\Scripts;{os.environ["FFMPEG_PATH"]};{os.environ["PATH"]}'

# 执行Python脚本
app_to_run = 'stream/app.py'
backend_to_run = 'stream/backend.py'
command_app = [os.environ['PYTHON_EXECUTABLE'], app_to_run]
command_backend = [os.environ['PYTHON_EXECUTABLE'], backend_to_run]


print("开始启动")
subprocess.Popen(command_backend)
subprocess.Popen(command_app)



