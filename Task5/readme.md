

# 任务5

## 数字人交互问答
任务描述：基于LLM，实现数字人的语音交互。

 
## 目录

- [上手指南](#上手指南)
  - [开发前的配置要求](#开发前的配置要求)
  - [安装步骤](#安装步骤)
- [文件目录说明](#文件目录说明)
- [开发的架构](#开发的架构)
- [部署](#部署)
- [使用到的框架](#使用到的框架)
- [贡献者](#贡献者)
  - [如何参与开源项目](#如何参与开源项目)
- [版本控制](#版本控制)
- [作者](#作者)
- [鸣谢](#鸣谢)

### 上手指南

###### 开发前的配置要求

1. cuda 12.1
2. python >= 3.10
3. 需要本地安装ollama，并下载qwen2:

    ollama qwen2

###### **安装步骤**

1.安装需要的库

     pip install -r requirements.txt

2.下载预训练ASR模型

    python downloader
    
3.后端运行ASR的api

    python utils/start_asr_service.py

4.运行gradio

    python app.py

### Todo List
已完成
 

 - 基础构建

未完成
 - [ ] RAG
 - [ ] Langgraph构建
 - [ ] Function Call自动调用其他API

### 文件目录说明
eg:

```
filetree 
│  app.py
│  config.json
│  list.txt
│  requirements.txt    
├─ASR
│  │  .msc
│  │  .mv
│  │  am.mvn
│  │  chn_jpn_yue_eng_ko_spectok.bpe.model
│  │  config.yaml
│  │  configuration.json
│  │  model.pt
│  │  README.md
│  │  
│  ├─._____temp
│  │  ├─example
│  │  └─fig
│  ├─example
│  │      en.mp3
│  │      ja.mp3
│  │      ko.mp3
│  │      yue.mp3
│  │      zh.mp3
│  │      
│  └─fig
│          aed_figure.png
│          asr_results.png
│          inference.png
│          sensevoice.png
│          ser_figure.png
│          ser_table.png
│          
├─log
│      log-2024-7-28.txt
│      README.md
│      
├─out
│      2.wav
│      openai_tts_2.wav
│      openai_tts_3.wav
│      
└─utils
    │  asr_client.py
    │  chatgpt_new.py
    │  common.py
    │  config.py
    │  logger.py
    │  start_asr_service.py
    │  test.wav
    │  tts_new.py
    │  __init__.py

```






### 使用到的框架

- [xxxxxxx](https://getbootstrap.com)
- [xxxxxxx](https://jquery.com)
- [xxxxxxx](https://laravel.com)

### 贡献者

请阅读**CONTRIBUTING.md** 查阅为该项目做出贡献的开发者。







### 作者

yafengli@mail.ustc.edu.cn
  

 *您也可以在贡献者名单中参看所有参与该项目的开发者。*

### 版权说明

该项目签署了MIT 授权许可，详情请参阅 [LICENSE.txt](https://github.com/shaojintian/Best_README_template/blob/master/LICENSE.txt)

### 鸣谢


- [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
- [Img Shields](https://shields.io)
- [Choose an Open Source License](https://choosealicense.com)
- [GitHub Pages](https://pages.github.com)
- [Animate.css](https://daneden.github.io/animate.css)
- [xxxxxxxxxxxxxx](https://connoratherton.com/loaders)

<!-- links -->
[your-project-path]:shaojintian/Best_README_template
[contributors-shield]: https://img.shields.io/github/contributors/shaojintian/Best_README_template.svg?style=flat-square
[contributors-url]: https://github.com/shaojintian/Best_README_template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/shaojintian/Best_README_template.svg?style=flat-square
[forks-url]: https://github.com/shaojintian/Best_README_template/network/members
[stars-shield]: https://img.shields.io/github/stars/shaojintian/Best_README_template.svg?style=flat-square
[stars-url]: https://github.com/shaojintian/Best_README_template/stargazers
[issues-shield]: https://img.shields.io/github/issues/shaojintian/Best_README_template.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/shaojintian/Best_README_template.svg
[license-shield]: https://img.shields.io/github/license/shaojintian/Best_README_template.svg?style=flat-square
[license-url]: https://github.com/shaojintian/Best_README_template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/shaojintian
