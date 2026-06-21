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
from tavily import TavilyClient
from IPython.display import Image, display
import operator
import re

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

        self.tools = {
            tool.name: tool
            for tool in tools
        }
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

            tool_name = t["name"]

            if tool_name not in self.tools:

                print("bad tool name")
                result = "bad tool name"

            else:

                result = self.tools[tool_name].invoke(
                    t["args"]
                )

            results.append(
                ToolMessage(
                    tool_call_id=t["id"],
                    name=tool_name,
                    content=str(result)
                )
            )

        print("Back to model")

        return {"messages": results}

prompt = """
    Você é um assistente de pesquisa inteligente. Use o mecanismo de busca para procurar informações. \
    Você tem permissão para fazer múltiplas chamadas (seja em conjunto ou em sequência) \
    Procure informações apenas quando tiver certeza do que quer \
    Se precisar pesquisar alguma informação antes de fazer a pergunta de acompanhamento, você tem permissão para fazer isso!
"""

query_passado = """"
    Qual país sediou a Copa do Mundo de futebol em 1998? Quem foi o campeão e qual foi o placar final?
    Qual era o Produto Interno Bruto (PIB) desse país no ano da Copa e qual o PIB atual (últimos dados disponíveis como 2023 ou 2024)
    Qual a capital desse país e qual a sua moeda atual? Responda cada pergunta separadamente
"""
messages_passado = [
    HumanMessage(content=query_passado)
]



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

messages = [
    HumanMessage(content="Como está o tempo em São Paulo e no Rio de Janeiro hoje?")
]

print("\n Iniciando interação do agente sobre perguntas do passado")

current_state = {}
### .stream para trazer o passo a passo do que foi realizado pelo agente
for s in abot.graph.stream({"messages": messages_passado}):
    current_state.update(s)
    print(s)
    print("---")

print("\nResultado final passado")
if current_state and 'llm' in current_state and current_state['llm']['messages']:
    print(current_state['llm']['messages'][-1].content)
else:
    print("Nenhum resultado final ou resultado inesperado")




print("Iniciando interação com o agente")
final_result_state = None


### Função Paralela
result = abot.graph.invoke({"messages": messages})
print("\nResultado final")
print(result['messages'][-1].content)


### .stream para trazer o passo a passo do que foi realizado pelo agente
for s in abot.graph.stream({"messages": messages}):
    print(s)
    print("---")
    final_result_state = s

print("\nResultado final")

if final_result_state and 'llm' in final_result_state and final_result_state['llm']['messages']:
    print(final_result_state['llm']['messages'][-1].content)
else:
    print("Nenhum resultado final ou resultado inesperado")



try: 
    image_data = abot.graph.get_graph().draw_mermaid_png()
    display(Image(data=image_data))
except Exception as e:
    print(f"Erro ao tentar gerar PNG no Mermaid {e}")


print("============================")
print("============================")
print("============================")
print("============================")
print("")


###### BUSCA REGULAR E BUSCA AGÊNTICA --- WEB SCRAPING ##########

## Busca regular -- Busca no Google, Edge
## Busca agêntica -- utiliza agentes autonomos para realizar a pesquisa
# Na busca agentica ela entende o contexto e ela é proativa, vai aprendendo conforme voce for executando


client = TavilyClient(api_key=TAVILY_API_KEY)

cidade = "Belém do Pará"
tavily_query = f"restaurante em {cidade} tripadvisor com maior quantidade de reviews e faixa de preço"

print("Iniciando busca agêntica por URL's do TripAdvisor com Tavily")

tripadvisor_url = None
try:
    tavily_results = client.search(query=tavily_query, max_results=5)
    if tavily_results and tavily_results["results"]:
        print(f"Tavily encontrou {len(tavily_results['results'])} resultado. Analisando...")
        for result in tavily_results["results"]:
            url = result['url']

            if "tripadvisor.com" in url or "tripadvisor.com.br" in url:
                tripadvisor_url = url
                break
        if not tripadvisor_url:
            print("Nenhum URL interessante foi encontrado nos primeiros resultados")
    else:
        print("Tavily não encontrou resultado para busca agentica")

except Exception as e:
    print(f"Erro na busca agêntica com Tavily {e}. Verifique chave de API ou conexão")

if tripadvisor_url:
    clean_url = re.sub(r'-oa\d+-', '-', tripadvisor_url)
    tripadvisor_url = clean_url
    print("URL encontrada limpa de paginação")

print("="*50)
print(f"URL final para a raspagem {tripadvisor_url if tripadvisor_url else 'NÃO ENCONTRADO'}")

    
