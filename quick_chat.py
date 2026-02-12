from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# Initializing the model
llm = ChatOllama(model="llama3.1", temperature=0.2)

# Invoking the model with a list of messages
res = llm.invoke([
    SystemMessage(content="You are helpful"),
    HumanMessage(content="Who is faheem_hamza on Instagram?")
])

print(res.content)
