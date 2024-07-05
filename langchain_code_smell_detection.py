import os
import re
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 确保设置了 OpenAI API 密钥
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# 创建 LLM
llm = OpenAI(temperature=0, model_name="gpt-4o")

# 定义评分提示模板
scoring_template = """
你是一个专业的内容评审员。请根据检查项 '{checkitem}' 对给定的 Markdown 内容进行评分：

内容：
{content}

请为该部分内容根据检查项 '{checkitem}' 打分（1-10分），并给出简短解释。
请输出结果，格式如下：

分数: 
依据：
"""

scoring_prompt = PromptTemplate(
    input_variables=["checkitem", "content"], template=scoring_template
)

# 创建评分链
scoring_chain = LLMChain(llm=llm, prompt=scoring_prompt)

def split_markdown(file_path):
    """按二级标题分割 Markdown 文件，并限制每部分的大小"""
    with open(file_path, "rt", encoding="utf-8") as file:
        content = file.read()

    # # 使用正则表达式分割文本
    # parts = re.split(r"(?=## )", content)

    # 使用正则表达式按缩进分割代码
    pattern = r"(?P<type>class|def)\s+(?P<name>\w+)\s*(\(.*?\))?\s*:(?P<body>.*?)(?=class\s+\w+|def\s+\w+|$)"

    parts = re.split(pattern, content)

    # 限制每部分的大小
    limited_parts = [part[:5000] for part in parts]

    return limited_parts

def parse_score(score_str):
    """解析从评分链返回的分数字符串，提取分数部分"""
    match = re.match(r"分数: (\d+)", score_str)
    if match:
        return int(match.group(1))
    return 0

def score_document_parts(file_path, checklist):
    """对文档的各个部分根据每个检查项进行评分，并计算总分"""
    document_parts = split_markdown(file_path)
    total_score = 0
    scores = []

    for i, part in enumerate(document_parts):
        scores.append(f"Part {i+1}:\n")
        part_scores = []
        for checkitem in checklist:
            score_output = scoring_chain.run(checkitem=checkitem, content=part.strip())
            part_scores.append(f"{checkitem}: {score_output}")

            # 解析分数并累计到总分
            score = parse_score(score_output)
            total_score += score

        scores.append("\n".join(part_scores))

    # 添加总分汇总
    scores.append(f"\nTotal Score: {total_score}")

    return "\n\n".join(scores)

def save_scores(file_path, scores):
    """将评分结果保存到新文件"""
    base_name = os.path.basename(file_path)
    name_without_ext = os.path.splitext(base_name)[0]
    output_file = f"{name_without_ext}_scores.txt"

    with open(output_file, "wt", encoding="utf-8") as file:
        file.write(scores)

    return output_file

def main():
    # 示例检查表
    checklist = [
        "不好的命名",
        "重复代码",
        "过长的函数或方法",
        "缺乏注释和文档",
        "神奇数字（Magic Numbers）",
        "复杂的条件和循环",
        "未处理的异常",
        "无效的或多余的代码",
        "低效的实现",
        "不一致的代码风格",
    ]

    # 对多个文件进行评分
    files = ["doc.md", "doc_1.md", "doc_2.md"]

    for file in files:
        print(f"Scoring {file}:")
        result = score_document_parts(file, checklist)
        output_file = save_scores(file, result)
        print(f"Scores saved to {output_file}")
        print("\n" + "=" * 50 + "\n")

if __name__ == "__main__":
    main()
