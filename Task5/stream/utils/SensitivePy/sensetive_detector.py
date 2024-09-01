from flashtext import KeywordProcessor
import os
current_path = os.path.dirname(os.path.abspath(__file__))

class KeywordFilter:
    def __init__(self):
        self.keyword_processor = KeywordProcessor()
    def load_keywords_from_file(self, filepath=os.path.join(current_path, 'words.txt')):
        """从文件中加载敏感词列表"""
        keywords = []
        try:
            with open(filepath, 'r', encoding='utf-8') as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        keywords.append(line)
        except IOError as e:
            print(f"Error reading file {filepath}: {e}")
            return
        
        for keyword in keywords:
            self.keyword_processor.add_keyword(keyword, '<sensetive word!>')
    
    def filter_text(self, text):
        """过滤并替换文本中的敏感词"""
        # detected_keywords = set()
        
        # 替换敏感词
        filtered_text = self.keyword_processor.replace_keywords(text)
        print('检测后的文本：', filtered_text)
        return filtered_text

if __name__ == "__main__":
    # 初始化敏感词过滤器
    keyword_filter = KeywordFilter()

    # 从文件中加载敏感词
    filepath = os.path.join(current_path, 'words.txt')
    keyword_filter.load_keywords_from_file(filepath)

    # 示例文本
    text = "我的密码是123456，身份证号码是123456789012345678，工资是10000元，家庭住址在上海，薪资待遇不错，供应商是ABC公司。"
    print(text)

    # 过滤并替换文本中的敏感词
    filtered_text = keyword_filter.filter_text(text)

    print("Filtered Text:", filtered_text)
