from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from agent.history import AgentHistory
from tools.tools import user_dict, UserData, tools
from agent.prompts import system_prompt
from utils.utils import handle_tool_calls, remove_think_tags, trim_latest_messsages


class Agent:
    def __init__(self, name: str, instructions: str, history: AgentHistory):
        self.__name = name
        self.__history = history

        self.__system_prompt = ChatPromptTemplate([('system', system_prompt)]).format_messages(name=name, instructions=instructions)

        user_dict[name] = UserData(instructions=instructions)

        self.__llm = ChatOllama(model="qwen3:8b").bind_tools(tools)

    @property
    def name(self) -> str:
        return self.__name

    def receive_message(self):
        history = self.__system_prompt + self.__history.get_llm_messages_based_on(self.__name)
        history = trim_latest_messsages(history, 20)
        response = handle_tool_calls(self.__llm, history, self.__llm.invoke(history))

        response_text = remove_think_tags(response.text())
        if response_text:
            self.__history.add(self.__name, remove_think_tags(response.text()))
