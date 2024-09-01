import json, logging, os, sys
from gradio_client import Client
import traceback
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_path)

from stream.utils.common import Common
import edge_tts

def edgetts(messages, stream=False):
    try:    
        common = Common()
        VOICE = "zh-CN-XiaoxiaoNeural"
        TEXT = messages
        # 初始化文本转语音的通信对象
        communicate = edge_tts.Communicate(TEXT, VOICE)
        if stream == True:
            response = communicate.stream()
            yield response
        # 确定输出文件名称
        file_path = os.path.join(current_path, 'voice_tmp')
        os.makedirs(file_path, exist_ok=True)
        file_name = 'edge_tts' + f'{common.get_bj_time(4)}.wav'
        voice_tmp_path = common.get_new_audio_path(file_path, file_name)
        # 写入文件
        communicate.save_sync(voice_tmp_path)
        return voice_tmp_path
    
    except Exception as e:
        logging.error(traceback.format_exc())
        logging.error(f'OpenAI_TTS请求失败: {e}')
        return None
    



if __name__ == '__main__':
    print(edgetts("你好"))