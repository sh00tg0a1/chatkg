# import os
import openai

# openai.organization = "org-5Xwd7o1U7UN1JBZIOu1sb6mz"
# openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.organization = "org-5Xwd7o1U7UN1JBZIOu1sb6mz"
# openai.api_key = "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"

OPENAI_API_KEY = "5cc0b8aef94b47db86c5b8c7f61ab9f2"
OPENAI_ENDPOINT  = "https://anyshare-demo-chatgpt.openai.azure.com/"
OPENAI_API_VERSION  = "2023-03-15-preview"
OPENAI_API_TYPE  = "azure"
openai.api_type = OPENAI_API_TYPE
openai.api_base = OPENAI_ENDPOINT
openai.api_version = OPENAI_API_VERSION
openai.api_key = OPENAI_API_KEY

class OpenAIChat():
    """OpenAIChat"""
    def __init__(self,
                 template="""\"\"\"{text}\"\"\"""",
                 text="Say Hello",
                 model="gpt-3.5-turbo") -> None:
        self._text = text
        self._template = template
        self._prompt = template.format(text)
        self._model = model

    # def set_text(self, text):
    #     '''
    #     text: 要设置的文本
    #     '''
    #     self._prompt = template.format(text)
    #     self._prompt = text

    def get_completion_with_text(self, text):
        '''
        text 基于指定模版输入的问题
        '''
        messages = [{"role": "user", "content": self._template.format(text)}]
        response = openai.ChatCompletion.create(
            # model=self._model,
            engine="aschatgpt35",
            messages=messages,
            temperature=0.9, # 模型输出的温度系数，控制输出的随机程度
        )
        # 调用 OpenAI 的 ChatCompletion 接口
        return response.choices[0].message["content"]

    def get_completion(self):
        '''
        prompt: 对应的提示
        model: 调用的模型，默认为 gpt-3.5-turbo(ChatGPT)，有内测资格的用户可以选择 gpt-4
        '''
        messages = [{"role": "user", "content": self._prompt}]
        response = openai.ChatCompletion.create(
            engine="aschatgpt35",
            # model=self._model,
            messages=messages,
            temperature=0, # 模型输出的温度系数，控制输出的随机程度
        )
        # 调用 OpenAI 的 ChatCompletion 接口
        return response.choices[0].message["content"]


def get_completion(prompt, model="gpt-3.5-turbo"):
    '''
    prompt: 对应的提示
    model: 调用的模型，默认为 gpt-3.5-turbo(ChatGPT)，有内测资格的用户可以选择 gpt-4
    '''
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # 模型输出的温度系数，控制输出的随机程度
    )
    # 调用 OpenAI 的 ChatCompletion 接口
    return response.choices[0].message["content"]


if __name__ == "__main__":
    prompt1 = """
    请生成包括书名、作者和类别的三本虚构书籍清单，并以 JSON 格式提供，其中包含以下键:book_id、title、author、genre。
    """

    # chat1 = OpenAIChat(prompt=prompt1)
    # print(chat1.get_completion())

    # 有步骤的文本
    text_1 = """
    泡一杯茶很容易。首先，需要把水烧开。
    在等待期间，拿一个杯子并把茶包放进去。
    一旦水足够热，就把它倒在茶包上。
    等待一会儿，让茶叶浸泡。几分钟后，取出茶包。
    如果你愿意，可以加一些糖或牛奶调味。
    就这样，你可以享受一杯美味的茶了。
    """

    template = """
    您将获得由三个引号括起来的文本。
    如果它包含一系列的指令，则需要按照以下格式重新编写这些指令：

    第一步 - ...
    第二步 - …
    …
    第N步 - …

    如果文本中不包含一系列的指令，则直接写“未提供步骤”。"
    \"\"\"{}\"\"\"
    """

    chat2 = OpenAIChat(template=template, text=text_1)
    # print(chat2.get_completion())

    # 无步骤的文本
    text_2 = """
    今天阳光明媚，鸟儿在歌唱。
    这是一个去公园散步的美好日子。
    鲜花盛开，树枝在微风中轻轻摇曳。
    人们外出享受着这美好的天气，有些人在野餐，有些人在玩游戏或者在草地上放松。
    这是一个完美的日子，可以在户外度过并欣赏大自然的美景。
    """

    prompt = """
    您将获得由三个引号括起来的文本。
    如果它包含一系列的指令，则需要按照以下格式重新编写这些指令：

    第一步 - ...
    第二步 - …
    …
    第N步 - …

    如果文本中不包含一系列的指令，则直接写“未提供步骤”。"
    \"\"\"{}\"\"\"
    """
    
    chat3 = OpenAIChat(template=prompt, text=text_2)
    # print(chat3.get_completion())

    # 生成评论
    lamp_review_zh = """
    我需要一盏漂亮的卧室灯，这款灯具有额外的储物功能，价格也不算太高。\
    我很快就收到了它。在运输过程中，我们的灯绳断了，但是公司很乐意寄送了一个新的。\
    几天后就收到了。这款灯很容易组装。我发现少了一个零件，于是联系了他们的客服，他们很快就给我寄来了缺失的零件！\
    在我看来，Lumina 是一家非常关心顾客和产品的优秀公司！
    """

    prompt = """
    以下用三个反引号分隔的产品评论的情感是什么？

    评论文本: ```{}```
    """

    chat4 = OpenAIChat(template=prompt, text=lamp_review_zh)
    response = chat4.get_completion()
    print(response)
