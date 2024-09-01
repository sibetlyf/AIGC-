import json

class Config:
    # 单例模式
    # _instance = None
    config = None

    # def __new__(cls, *args, **kwargs):
    #     if not cls._instance:
    #         cls._instance = super(Config, cls).__new__(cls)  # 不再传递 *args, **kwargs
    #     return cls._instance

    def __init__(self, config_file):
        if self.config is None:
            with open(config_file, 'r', encoding="utf-8") as f:
                self.config = json.load(f)

    def get(self, *keys):
        result = self.config
        for key in keys:
            result = result.get(key, None)
            if result is None:
                break
        return result

# 字符串数组格式转换
def textarea_data_change(data):
    """
    字符串数组数据格式转换
    """
    tmp_str = ""
    for tmp in data:
        tmp_str = tmp_str + tmp + "\n"
    
    return tmp_str