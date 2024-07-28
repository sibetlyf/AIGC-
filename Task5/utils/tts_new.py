import json, logging, os
from gradio_client import Client
import traceback

from utils.common import Common
from utils.logger import Configure_logger
from utils.config import Config
import edge_tts


class MY_TTS:
    def __init__(self, config_path):
        self.common = Common()
        self.config = Config(config_path)

        # 请求超时
        self.timeout = 60

        # 日志文件路径
        file_path = "./log/log-" + self.common.get_bj_time(1) + ".txt"
        Configure_logger(file_path)

        try:
            self.audio_out_path = self.config.get("play_audio", "out_path")

            if not os.path.isabs(self.audio_out_path):
                if not self.audio_out_path.startswith('./'):
                    self.audio_out_path = './' + self.audio_out_path
        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error("请检查播放音频的音频输出路径配置！！！这将影响程序使用！")

    # 请求OpenAI_TTS的api
    # 暂时采用EdgeTTS
    def text_to_speech(self, data):
        try:
            VOICE = "zh-CN-XiaoxiaoNeural"
            TEXT = data['content']
            # 初始化文本转语音的通信对象
            communicate = edge_tts.Communicate(TEXT, VOICE)
            response = communicate.stream()
            # 确定输出文件名称
            file_name = 'openai_tts_' + self.common.get_bj_time(4) + '.wav'
            voice_tmp_path = self.common.get_new_audio_path(self.audio_out_path, file_name)
            # 写入文件
            communicate.save_sync(voice_tmp_path)
            # # 以二进制写模式打开输出文件
            # with open(voice_tmp_path, "wb") as file:
            #     # 遍历由文本转语音过程产生的数据块
            #     for chunk in communicate.stream():
            #         # 如果数据块类型为音频，则写入文件
            #         if chunk["type"] == "audio":
            #             file.write(chunk["data"])
            #         # 如果数据块类型为单词边界，则打印信息
            #         elif chunk["type"] == "WordBoundary":
            #             print(f"单词边界: {chunk}")
            return voice_tmp_path

        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error(f'OpenAI_TTS请求失败: {e}')
            return None

if __name__ == '__main__':
    config_path = "../config.json"
    config = Config(config_path)

    my_tts = MY_TTS(config_path)
    path = my_tts.text_to_speech("你好")
    print(path)