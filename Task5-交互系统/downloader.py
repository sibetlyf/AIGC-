#模型下载
from modelscope import snapshot_download

asr_model = snapshot_download('iic/SenseVoiceSmall', local_dir='./ASR')
tts_model = snapshot_download('iic/CosyVoice-300M-Instruct', local_dir='./tts/CosyVoice_For_Windows/pretrained_models')