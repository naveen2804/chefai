from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_ibm import ChatWatsonx
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
import os
import json

load_dotenv()

app = FastAPI()

llm = ChatWatsonx(
    model_id="openai/gpt-oss-120b",
    apikey=os.getenv("WATSONX_APIKEY"),
    project_id=os.getenv("PROJECT_ID"),
    url="https://us-south.ml.cloud.ibm.com",
    params={
        "decoding_method": "greedy",
        "max_new_tokens": 4096,
        "min_new_tokens": 0,
        "stop_sequences": [],
        "repetition_penalty": 1
    }
)


def generate_stream(topic: str, system_content: str):
    system_message = SystemMessage(content=system_content)
    human_message = HumanMessage(content=topic)

    for chunk in llm.stream([system_message, human_message]):
        yield json.dumps({"content": chunk.content}) + "\n"


@app.get("/stream")
async def stream_response(topic: str, system_content: str):
    """
    Pass the topic in the query parameter, e.g. /stream?topic=moon
    """
    return StreamingResponse(
        generate_stream(topic,system_content),
        media_type="text/event-stream"
    )
