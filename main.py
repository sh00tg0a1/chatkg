from fastapi import FastAPI
from apps.search_doc import qa_with_index, qa_without_index, get_faiss


app = FastAPI()


def get_res_format(message):
    return {
        "res": message,
    }


@app.get("/question_type/")
async def question_type(question: str, with_index: bool = False):
    if with_index:
        index = get_faiss()
        return get_res_format(qa_with_index(question, index))

    return get_res_format(qa_without_index(question))
