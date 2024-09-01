import sys, os
import requests, json
import pandas as pd

os.environ['COZE_API_TOKEN'] = 'pat_4AyvTNR58wBDZIm3NyHZNCG8k4W7y36lRsbcvJnQRkJ5W06FcurX8cQmL0lgwBhW'
os.environ['COZE_BOT_ID'] = "7407371111056703524"

class Coze:
    def __init__(self,
                 bot_id=None,
                 api_token=None,
                 max_chat_rounds=20,
                 stream=True,
                 history=None):
        self.bot_id = os.environ['COZE_BOT_ID']
        self.api_token = api_token if api_token else os.environ['COZE_API_TOKEN']
        self.history = [] if history is None else history
        self.max_chat_rounds = max_chat_rounds
        self.stream = stream
        self.url = 'https://api.coze.cn/open_api/v2/chat'
        self.headers = {
            'Authorization': f'Bearer {self.api_token}',
            'Content-Type': 'application/json',
            'Accept': '*/*',
            'Host': 'api.coze.cn',
            'Connection': 'keep-alive'
        }

    @classmethod
    def build_messages(cls, history=None):
        messages = []
        history = history if history else []
        for prompt, response in history:
            pair = [{"role": "user", "content": prompt, "content_type": "text"},
                    {"role": "assistant", "content": response}]
            messages.extend(pair)
        return messages

    @staticmethod
    def get_response(messages):
        dfmsg = pd.DataFrame(messages)
        dftool = dfmsg.loc[dfmsg['type'] == 'function_call']
        for content in dftool['content']:
            info = json.loads(content)
            s = 'call function: ' + str(info['name']) + '; args =' + str(info['arguments'])
            print(s, file=sys.stderr)

        dfans = dfmsg.loc[dfmsg['type'] == 'answer']
        if len(dfans) > 0:
            response = ''.join(dfans['content'].tolist())
            print(response)  # 直接打印响应到控制台
        else:
            response = ''
        return response

    def chat(self, query, stream=False):
        data = {
            "conversation_id": "123",
            "bot_id": self.bot_id,
            "user": "user",
            "query": query,
            "stream": stream,
            "chat_history": self.build_messages(self.history)
        }
        json_data = json.dumps(data)
        result = requests.post(self.url, headers=self.headers, data=json_data, stream=data["stream"])

        if not data["stream"] and result.status_code == 200:
            dic = json.loads(result.content.decode('utf-8'))
            response = self.get_response(dic['messages'])

        elif data['stream'] and result.status_code == 200:
            messages = []
            for line in result.iter_lines():
                if not line:
                    continue
                try:
                    line = line.decode('utf-8')
                    line = line[5:] if line.startswith('data:') else line
                    dic = json.loads(line)
                    if dic['event'] == 'message':
                        messages.append(dic['message'])
                    response = self.get_response(messages)
                except Exception as err:
                    print(err)
                    break
        else:
            print(f"request failed, status code: {result.status_code}")
        result.close()
        return response

    def __call__(self, query):
        len_his = len(self.history)
        if len_his >= self.max_chat_rounds + 1:
            self.history = self.history[len_his - self.max_chat_rounds:]
        response = self.chat(query, stream=self.stream)
        self.history.append((query, response))
        return response

if __name__ == "__main__":
    chat = Coze(api_token=os.environ['COZE_API_TOKEN'],
                bot_id=os.environ['COZE_BOT_ID'],
                max_chat_rounds=20,
                stream=True)
    response = chat.chat('你好')