# 移动鸿鹄第三小组——基于AIGC技术的交互式数字人

## 任务5
 数字人交互问答
任务描述：基于LLM，实现数字人的语音交互。

 
## 目录
- [系统介绍](#系统介绍)
- [上手指南](#上手指南)
  - [开发前的配置要求](#开发前的配置要求)
  - [安装步骤](#安装步骤)
- [功能说明](#功能说明)
- [文件目录说明](#文件目录说明)
- [使用到的框架](#使用到的框架)
- [特别感谢](#特别感谢)
  -  [梧桐鸿鹄训练营](https://it.10086.cn/honghu/hhweb/#/home)
  - [童同老师]
- [作者](#作者)
- [版权说明](#版权说明)

## 系统介绍
开源语音交互项目以gradio为前端，通过requests库向后端发送请求，接受回复后进行展示，支持各种openai格式模型的调用。开发人员可以直接将框架集成到数字人项目中，从而实现完整的语音交互功能。除此之外，该项目还提供了语音复刻与自定义角色功能，能够实现更为精准与个性化的互动流程。针对不熟悉Python的用户，我们还提供了预设环境与一键运行脚本，方便快速体验智能交互流程。欢迎随时使用！



### 上手指南

##### 开发前的配置要求
（如果选择打包好的py310文件可忽略）

1. cuda 12.1
2. python >= 3.10
3. ~~需要本地安装ollama，并下载qwen2
    ollama qwen2~~
    需要本地模型支持openai key格式的调用
    默认配置为本地运行的ollama api

#### **安装步骤**
方法一：

 - 安装需要的库

	     pip install -r requirements.txt

- 下载预训练模型（sensevocie + cosyvoice）
	sensevoice-small模型保存在"/ASR"文件夹，cosyvoice模保存在"tts/CosyVoice_For_Windows/pretrained_models"文件夹

	    python downloader
    
- 启动演示app同时运行后端api与前端gradio）

	    python ./run_app.py

方法二：下载配置环境，并直接运行main.bat文件或run.sh文件

	   mian.bat
基础交互界面如图所示
![gradio_interface](https://github.com/sibetlyf/AIGC-/blob/b08217825087201d9c4614601b65d139ff6a34a2/pictures/interface.png)
### 功能说明
#### 系统基本结构
![系统结构图](https://github.com/sibetlyf/AIGC-/blob/main/pictures/framework.png)

#### 系统功能说明
##### 1.流式交互模式
![页面1](https://github.com/sibetlyf/AIGC-/blob/main/pictures/interface.png)
 1. 点击提交音频文件或麦克风录制语音输入音频，系统会自动将语音诸转为文本并填写在“输入内容”框中。
 2. 确认文本无误后，点击提交按钮，系统自动生成流式回复文本，并进行流式语音输出。
 3. (待完善）已经利用mimicmotion提取了一些主持人相关的动作（面部+躯体+手部姿势），未来会加入根据输入状态自动播放倾听/发言/挥手道别动作的功能
 
 ##### 2.情感控制模式
 ![页面2](https://github.com/sibetlyf/AIGC-/blob/main/pictures/instruct.png)
 1. 提交需要阅读的文本，并选择喜欢的音色
 2. 模型会返回加入语气词标签的文本与对应的instruct情感控制文本（该部分目前由字节豆包模型完成，Qwen2：7B无法满足语气词标签生成的要求）。
 3. 点击提交按钮，即可生成音频。
##### 3.RAG模式
![页面3](https://github.com/sibetlyf/AIGC-/blob/main/pictures/RAG.png)
1. 上传需要检索的pdf，点击提交按钮，生成本地faiss向量库，向量库地址会自动填写。如果有本地已建立好的向量库，也可以直接填写地址。（OCR时间较长，需要等待）
2. 点击提交按钮，完成对话
##### 4.多角色模式（待完善）
![页面4](https://github.com/sibetlyf/AIGC-/blob/main/pictures/RouterChain.png)
1. 默认包含三个角色（医生、秘书、英语教师），可以点击角色创建按钮创建自定义角色。
2. 选择需要的角色进行提问，Router Chain会自动调用最合适的角色。（目前qwen2：7B有可能会错误调用角色，待完善）
##### 5.语音复刻功能整合
![页面5](https://github.com/sibetlyf/AIGC-/blob/main/pictures/VoiceClone.png)
1. 输入prompt音频，并点击识别，识别后的文本会展示在文本框中。如果识别结果有误差，需要手动调节。（prompt音频与prompt文本内容需要一致！！！，音频不能超过30s）
2. 输入需要阅读的文本。
3. 点击生成复刻音频，即可进行语音复刻。
4. 点击保存音色，将复刻后的语音控制模型保存在本地。
5. 点击刷新按钮，即可使用复刻音色。
##### 6.配置页面
![配置页](https://github.com/sibetlyf/AIGC-/blob/main/pictures/settings.png)
1. 对称加密密钥管理
2. 对话模型API配置（默认为ollama）
3. 点击保存配置即可。



已完成
 - 基础构建
 - 语音复刻
 - 动作复刻
 - 情感预测
 - RAG与多角色模式

未完成
 - [ ] 口唇驱动
 - [ ] 动作播放
 - [ ] Function Call功能
 - [ ] 形象生成功能与动作复刻功能整合
 - [ ] 语音调用直接调用Agent功能

### 文件目录说明
eg:

```
filetree 
├── LICENSE.txt
├── README.md
├── /ASR/
│  ├── ……
├── /tts/
│  ├── /CosyVoice_For_Windows/
│  │  ├── /third_party/
│  │  └── stream_tts.py
│  │  └── ……
├── /stream/
│  ├── app.py
│  └── backend.py
│  └── ……
├── /py310/
├── downloader.py
├── run_app.py
├── main.bat
├── run_app.bat
└── run_backend.bat

```


### 使用到的框架


- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice)
- [CosyVocie](https://github.com/FunAudioLLM/CosyVoice)
- [Mimicmotion](https://github.com/Tencent/MimicMotion)
- [Gradio](https://github.com/gradio-app/gradio)
- [LangChain](https://github.com/langchain-ai/langchain)
- [ms-Swift](https://github.com/modelscope/ms-swift)

### 特别感谢

- [梧桐鸿鹄训练营](https://it.10086.cn/honghu/hhweb/#/home)为本项目提供的指导







### 作者
- 李亚峰 yafengli@mail.ustc.edu.cn
- 钟建梅
- 饶季勇
- 陶能坤
  


### 版权说明

该项目签署了MIT 授权许可，详情请参阅 [LICENSE.txt](https://github.com/sibetlyf/AIGC-/blob/main/LICENSE)