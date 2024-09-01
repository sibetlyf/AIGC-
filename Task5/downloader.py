#模型下载
from modelscope import snapshot_download

print(f"开始下载sensevoice模型..., 保存路径./ASR")
asr_model = snapshot_download('iic/SenseVoiceSmall', local_dir='./ASR')
print(f"开始下载tts模型..., 保存路径./tts/CosyVoice_For_Windows/pretrained_models/CosyVoice-300M-Instruct")
tts_model = snapshot_download('iic/CosyVoice-300M-Instruct', local_dir='./tts/CosyVoice_For_Windows/pretrained_models/CosyVoice-300M-Instruct')
print(f"开始下载rag模型..., 保存路径stream\RAG\indexer_model\zpoint_large_embedding_zh")
indexer_model = snapshot_download('maple77/zpoint_large_embedding_zh', local_dir='stream\RAG\indexer_model\zpoint_large_embedding_zh')