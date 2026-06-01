import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.messages import BaseMessage
from langchain_community.tools import TavilySearchResults
from langchain_tavily import TavilySearch
from IPython.display import Image, display
import operator

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

tool = TavilySearch(max_results=4)


class AgentStates(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class Agents:

    def __init__(self, model, tools, system=""):
        self.system = system
        self.model = model
        self.tools = tools
        graph = StateGraph(AgentStates)
        graph.add_node("llm", self.call_gpt)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exist_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")

        self.graph = graph.compile()

    def exist_action(self, state:AgentStates):
        result = state["messages"][-1]
        return len(result.tool_calls) > 0

    def call_gpt(self, state:AgentStates):
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages' : [message]}


    def take_action(self, state:AgentStates):
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling {t}")
            if not t.name in self.tools:
                print("\n ... bad tool name...")
                result = "bad tool name, retry"
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model")
        return {'messages': results}

prompt = """
    Você é um assistente de pesquisa inteligente. Use o mecanismo de busca para procurar informações. \
    Você tem permissão para fazer múltiplas chamadas (seja em conjunto ou em sequência) \
    Procure informações apenas quando tiver certeza do que quer \
    Se precisar pesquisar alguma informação antes de fazer a pergunta de acompanhamento, você tem permissão para fazer isso!
"""

model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
).bind_tools([tool])

abot = Agents(
    model=model,
    tools=[tool],
    system=prompt
)

mermaid_code = abot.graph.get_graph().draw_mermaid()
print(mermaid_code)


try: 
    image_data = abot.graph.get_graph().draw_mermaid_png()
    display(Image(data=image_data))
except Exception as e:
    print(f"Erro ao tentar gerar PNG no Mermaid {e}")
    
