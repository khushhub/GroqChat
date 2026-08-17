import os
from operator import itemgetter
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages
from langchain_groq import ChatGroq

# 1. Load environment variables & setup LangSmith Tracing
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Full-Featured-Chatbot")

# 2. Initialize Model & Parser
groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="openai/gpt-oss-120b", groq_api_key=groq_api_key)
parser = StrOutputParser()

# 3. Message Trimmer Configuration
trimmer = trim_messages(
    max_tokens=200,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human"
)

# 4. Prompt Template Construction
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer all questions to the best of your ability in {language}."),
    MessagesPlaceholder(variable_name="messages"),
])

# 5. LCEL Pipeline with Trimming
chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
    | parser
)

# 6. In-Memory Chat History Management
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Wrap with Message History
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
    history_messages_key="messages",
)

# 7. FastAPI App Definition & Middleware
app = FastAPI(
    title="Full-Featured Conversational Chatbot Server",
    version="1.0",
    description="LangChain LCEL Chatbot Server with InMemoryChatMessageHistory, LangServe, and LangSmith integration."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 8. Add LangServe Routes
add_routes(
    app,
    chain_with_history,
    path="/chatbot"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)