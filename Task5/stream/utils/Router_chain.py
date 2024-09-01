from langchain.chains.router import MultiPromptChain
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router.llm_router import RouterOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.conversation.base import ConversationChain
from langchain.chains.router.llm_router import LLMRouterChain
from langchain_community.chat_models.openai import ChatOpenAI
import os
import sys
import json
current_path = current_path = os.path.dirname(os.path.abspath(__file__))




# # 1.定义不同的子链的prompt模板
# ## 医生
# docter_template = """
# 你是一个医生，你需要根据用户健康咨询，给出一个完整的健康建议，包括问题，用户的回答，医生自己的回答，以及医生自己的建议。
# 如果你不知道,请说"建议咨询真正专业的医生"

# 下面是需要你来回答的问题：
# {input}

# """
# ## 秘书
# secretary_template = """
# 你是一个秘书，你需要根据用户提问，给出一个秘书教师身份的回答，并给出对应的日程安排建议。

# 下面是需要你来回答的问题：
# {input}

# """

# ## 教师
# english_teacher_template ="""
# 你是一个英语老师，用户输入的中文词汇，你需要提供对应的英文单词，包括单词词性，对应的词组和造句。

# 下面是需要你来回答的问题：
# {input}
# """

# # 2. 创建路由目录,记录子链名称/子链描述和子链prompt template,便于创建子链和用于路由的PromptTemplate
# prompt_infos = [
#     {
#         "name": "doctor",
#         "description": "专业的健康咨询助手，可以回答用户的健康问题",
#         "prompt_template": docter_template,
#     },
#     {
#         "name": "secretary",
#         "description": "负责用户的日程安排和公务处理，可以回答用户的日程安排问题与公务问题",
#         "prompt_template": secretary_template,
#     },
#     {
#         "name": "english teacher",
#         "description": "英语老师，可以解答用户关于英语和教育方面的问题",
#         "prompt_template": english_teacher_template,

#     },
# ]

# # 3.存储对应的子链名称和描述,为创建MULTI_PROMPT_ROUTER_TEMPLATE做准备
# destinations = [
#     f"{p['name']}: {p['description']}" for p in prompt_infos
# ]
# ## 将描述转换为字符串类型
# destinations_str = "\n".join(destinations)

# # 4.使用MULTI_PROMPT_ROUTER_TEMPLATE.format()进行格式化
# """
# 格式化完毕后生成一个字符串,作为prompt template
# """
# from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

# router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
#     destinations=destinations_str
# )
# print("路由模板:\n", router_template)

# # 5.生成prompt template实例,template中包含{input}作为输入,采用RouterOutputParser作为输出解析器
# from langchain.chains.router.llm_router import RouterOutputParser
# from langchain_core.prompts import PromptTemplate
# router_prompt = PromptTemplate(
#     template=router_template,
#     input_variables=["input"],
#     output_parser=RouterOutputParser(),
# )
# print(router_prompt)

# # 6.定义LLMRouterChain,用于路由和子链的调用
# from langchain.chains.router.llm_router import LLMRouterChain
# from langchain_community.chat_models.openai import ChatOpenAI
# import os

# ## 创建基底
# os.environ["OPENAI_API_KEY"] = "ollama"
# os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1/"

# llm = ChatOpenAI(
#             model="qwen2",
#             openai_api_key="ollama",
#             openai_api_base='http://localhost:11434/v1/',
#             stop=['<|im_end|>'],
#             streaming=True,
#             max_tokens=2048,
#         )  
# ## 创建router_chain路由链
# router_chain = LLMRouterChain.from_llm(
#                                     llm, 
#                                     router_prompt,  
#                                     verbose=True)

# # 7.定义子链/default chain
# from langchain.chains.llm import LLMChain
# from langchain.chains.conversation.base import ConversationChain

# ## 创建候选链,包含所有下游子链
# candadite_chains = {}
# ## 遍历路由目录,生成各子链并放入候选链字典
# for p_info in prompt_infos:
#     name = p_info["name"]
#     prompt_template = p_info["prompt_template"]
#     prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
#     chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
#     candadite_chains[name] = chain

# ## 生成默认链
# deafult_chain = ConversationChain(llm=llm, output_key="text",)

# # 8.构建多提示链 MultiPromptChain
# from langchain.chains.router import MultiPromptChain
# chain = MultiPromptChain(
#     router_chain=router_chain,
#     destination_chains=candadite_chains,
#     default_chain=deafult_chain,
#     verbose=True,
# )



class RouterChainManager():
    def __init__(self):
        self._last_doctor = None
        self._last_secretary = None
        self._last_english_teacher = None
        self.chain = None
        self.custom_agents_file = os.path.join(current_path, 'logs/custom_agents.json')
    
    def save_custom_agent(self, name, description, template):
        """保存用户创建的agent"""
        custom_agents = self.load_custom_agents()
        custom_agents[name] = {
            "description": description,
            "template": template, 
        }
        with open(self.custom_agents_file, 'w') as file:
            json.dump(custom_agents, file, indent=4)
    
    def load_custom_agents(self):
        """加载用户自定义的agent"""
        try:
            with open(self.custom_agents_file, 'r') as file:
                content = file.read().strip()  # 读取文件内容并去除首尾空白
                if not content:  # 如果文件内容为空
                    return {}
                return json.loads(content)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            print("Error decoding JSON from file")
            return {}
        except Exception as e:
            print(f"Error loading custom agents: {e}")
            return {}
    
    def delete_custom_agent(self, name):
        """删除指定名称的 custom_agent"""
        custom_agents = self.load_custom_agents()
        if name in custom_agents:
            del custom_agents[name]
            with open(self.custom_agents_file, 'w') as file:
                json.dump(custom_agents, file, indent=4)
            return True
        return False
    
    def ensure_input_placeholder(self, template):
        """确保占位符正确"""
        if "{input}" not in template:
            template += "\n下面是你需要回答的问题: \n{input}"
        return template
    
    def create_router_chain(self, llm=None, doctor=False, secretary=False, english_teacher=False, custom_agents=False):
        if (doctor != self._last_doctor or
            secretary != self._last_secretary or
            english_teacher != self._last_english_teacher or
            custom_agents != self.load_custom_agents()):

            # 更新所选的agent
            self._last_doctor = doctor
            self._last_secretary = secretary
            self._last_english_teacher = english_teacher

            # 加载自定义agent
            if custom_agents == True:
                custom_agents = self.load_custom_agents()

            # 1.定义不同的子链的prompt模板
            ## 医生
            doctor_template = """
            你是一个医生，你需要根据用户健康咨询，给出一个完整的健康建议，包括问题，用户的回答，医生自己的回答，以及医生自己的建议。
            如果你不知道,请说"建议咨询真正专业的医生"

            下面是需要你来回答的问题：
            {input}

            """
            ## 秘书
            secretary_template = """
            你是一个秘书，你需要根据用户提问，给出一个秘书教师身份的回答，并给出对应的日程安排建议。

            下面是需要你来回答的问题：
            {input}

            """

            ## 教师
            english_teacher_template ="""
            你是一个英语老师，用户输入的中文词汇，你需要提供对应的英文单词，包括单词词性，对应的词组和造句。

            下面是需要你来回答的问题：
            {input}
            """

            # 2. 创建路由目录,记录子链名称/子链描述和子链prompt template,便于创建子链和用于路由的PromptTemplate
            prompt_infos = []
            if doctor == True:
                prompt_infos.append({
                    "name": "doctor",
                    "description": "专业的健康咨询助手，可以回答用户的健康问题",
                    "prompt_template": doctor_template,
                })
            if secretary == True:
                prompt_infos.append({
                    "name": "secretary",
                    "description": "负责用户的日程安排和公务处理，可以回答用户的日程安排问题与公务问题",
                    "prompt_template": secretary_template,
                })
            if english_teacher == True:
                prompt_infos.append({
                    "name": "english teacher",
                    "description": "英语老师，可以解答用户关于英语和教育方面的问题",
                    "prompt_template": english_teacher_template,
                })

            # 加入自定义Agent
            if custom_agents:
                for name, data in custom_agents.items():
                    if name not in [p['name'] for p in prompt_infos]:
                        prompt_template = self.ensure_input_placeholder(data.get('template', ''))
                        prompt_infos.append({
                            "name": name,
                            "description": data.get('description', ''),
                            "prompt_template": prompt_template,
                        })

            # 3.存储对应的子链名称和描述,为创建MULTI_PROMPT_ROUTER_TEMPLATE做准备
            destinations = [
                f"{p['name']}: {p['description']}" for p in prompt_infos
            ]
            ## 将描述转换为字符串类型
            destinations_str = "\n".join(destinations)

            # 4.使用MULTI_PROMPT_ROUTER_TEMPLATE.format()进行格式化
            """
            格式化完毕后生成一个字符串,作为prompt template
            """
            router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
                destinations=destinations_str
            )
            print("路由模板:\n", router_template)

            # 5.生成prompt template实例,template中包含{input}作为输入,采用RouterOutputParser作为输出解析器
            router_prompt = PromptTemplate(
                template=router_template,
                input_variables=["input"],
                output_parser=RouterOutputParser(),
            )
            print(router_prompt)

            # 6.定义LLMRouterChain,用于路由和子链的调用
            ## 创建基底
            if llm == None:
                os.environ["OPENAI_API_KEY"] = "ollama"
                os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1/"

                llm = ChatOpenAI(
                            model="qwen2",
                            openai_api_key="ollama",
                            openai_api_base='http://localhost:11434/v1/',
                            stop=['<|im_end|>'],
                            streaming=True,
                            max_tokens=2048,
                        )  
            else:
                llm = llm
            ## 创建router_chain路由链
            router_chain = LLMRouterChain.from_llm(
                                                llm, 
                                                router_prompt,  
                                                verbose=True)

            # 7.定义子链/default chain
            ## 创建候选链,包含所有下游子链
            candidate_chains = {}
            ## 遍历路由目录,生成各子链并放入候选链字典
            for p_info in prompt_infos:
                name = p_info["name"]
                prompt_template = p_info["prompt_template"]
                prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
                chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
                candidate_chains[name] = chain

            ## 生成默认链
            default_chain = ConversationChain(llm=llm, output_key="text",)

            # 8.构建多提示链 MultiPromptChain
            chain = MultiPromptChain(
                router_chain=router_chain,
                destination_chains=candidate_chains,
                default_chain=default_chain,
                verbose=True,
            )
            self.chain = chain

        return self.chain


# def create_router_chain(self, llm=None):
#     # 1.定义不同的子链的prompt模板
#     ## 医生
#     docter_template = """
#     你是一个医生，你需要根据用户健康咨询，给出一个完整的健康建议，包括问题，用户的回答，医生自己的回答，以及医生自己的建议。
#     如果你不知道,请说"建议咨询真正专业的医生"

#     下面是需要你来回答的问题：
#     {input}

#     """
#     ## 秘书
#     secretary_template = """
#     你是一个秘书，你需要根据用户提问，给出一个秘书教师身份的回答，并给出对应的日程安排建议。

#     下面是需要你来回答的问题：
#     {input}

#     """

#     ## 教师
#     english_teacher_template ="""
#     你是一个英语老师，用户输入的中文词汇，你需要提供对应的英文单词，包括单词词性，对应的词组和造句。

#     下面是需要你来回答的问题：
#     {input}
#     """

#     # 2. 创建路由目录,记录子链名称/子链描述和子链prompt template,便于创建子链和用于路由的PromptTemplate
#     prompt_infos = [
#         {
#             "name": "doctor",
#             "description": "专业的健康咨询助手，可以回答用户的健康问题",
#             "prompt_template": docter_template,
#         },
#         {
#             "name": "secretary",
#             "description": "负责用户的日程安排和公务处理，可以回答用户的日程安排问题与公务问题",
#             "prompt_template": secretary_template,
#         },
#         {
#             "name": "english teacher",
#             "description": "英语老师，可以解答用户关于英语和教育方面的问题",
#             "prompt_template": english_teacher_template,

#         },
#     ]

#     # 3.存储对应的子链名称和描述,为创建MULTI_PROMPT_ROUTER_TEMPLATE做准备
#     destinations = [
#         f"{p['name']}: {p['description']}" for p in prompt_infos
#     ]
#     ## 将描述转换为字符串类型
#     destinations_str = "\n".join(destinations)

#     # 4.使用MULTI_PROMPT_ROUTER_TEMPLATE.format()进行格式化
#     """
#     格式化完毕后生成一个字符串,作为prompt template
#     """
#     router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
#         destinations=destinations_str
#     )
#     print("路由模板:\n", router_template)

#     # 5.生成prompt template实例,template中包含{input}作为输入,采用RouterOutputParser作为输出解析器
#     router_prompt = PromptTemplate(
#         template=router_template,
#         input_variables=["input"],
#         output_parser=RouterOutputParser(),
#     )
#     print(router_prompt)

#     # 6.定义LLMRouterChain,用于路由和子链的调用
#     ## 创建基底
#     if llm:
#         os.environ["OPENAI_API_KEY"] = "ollama"
#         os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1/"

#         llm = ChatOpenAI(
#                     model="qwen2",
#                     openai_api_key="ollama",
#                     openai_api_base='http://localhost:11434/v1/',
#                     stop=['<|im_end|>'],
#                     streaming=True,
#                     max_tokens=2048,
#                 )  
#     else:
#         llm = llm
#     ## 创建router_chain路由链
#     router_chain = LLMRouterChain.from_llm(
#                                         llm, 
#                                         router_prompt,  
#                                         verbose=True)

#     # 7.定义子链/default chain
#     ## 创建候选链,包含所有下游子链
#     candadite_chains = {}
#     ## 遍历路由目录,生成各子链并放入候选链字典
#     for p_info in prompt_infos:
#         name = p_info["name"]
#         prompt_template = p_info["prompt_template"]
#         prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
#         chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
#         candadite_chains[name] = chain

#     ## 生成默认链
#     deafult_chain = ConversationChain(llm=llm, output_key="text",)

#     # 8.构建多提示链 MultiPromptChain
#     from langchain.chains.router import MultiPromptChain
#     chain = MultiPromptChain(
#         router_chain=router_chain,
#         destination_chains=candadite_chains,
#         default_chain=deafult_chain,
#         verbose=True,
#     )
#     return chain

def main():
    router = RouterChainManager()
    # 保存自定义代理
    router.save_custom_agent(name="nutritionist", description="营养师，可以提供饮食建议", template="你是一个营养师，用户询问有关饮食的问题，请给出建议。")

    chain = router.create_router_chain(doctor=True, secretary=True, english_teacher=True, custom_agents=False)
    # chain = create_router_chain()
    ret = chain.stream({"input": "你好,请问如何减肥？"})
    text = ""
    for token in ret:
        token = token['text']
        js_data = {"code": "200", "msg": "ok", "data": token}
        text += token
        print(js_data)
        print("*********************")

if __name__ == "__main__":
    main()
