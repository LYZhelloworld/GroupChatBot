from typing import Iterable

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from agent.agent import Agent
from agent.history import AgentHistory
from orchestrator.prompts import system_prompt
from tools.tools import get_user_list
from utils.constants import *
from utils.utils import handle_tool_calls, remove_think_tags, trim_latest_messsages


class Orchestrator:
    def __init__(self, agents: Iterable[Agent], history: AgentHistory):
        self.__agents = list(agents)
        self.__agent_names = [agent.name for agent in self.__agents]
        self.__history = history

        self.__system_prompt = ChatPromptTemplate([('system', system_prompt)]).format_messages(
            user_list='\n'.join(self.__agent_names))
        self.__llm = ChatOllama(base_url=OLLAMA_URL, model=OLLAMA_MODEL).bind_tools([get_user_list])

    def chat(self, message: str):
        self.__history.add(USER_AGENT_NAME, message)

        while True:
            next_agent = self.__get_next_agent()
            if next_agent == USER_AGENT_NAME:
                break
            agent = self.__agents[self.__agent_names.index(next_agent)]
            agent.receive_message()

    def __generate_message_history(self):
        messages: list[BaseMessage] = []
        for item in self.__history.history:
            messages.append(AIMessage(content=item.name))
            messages.append(HumanMessage(content=f'[{item.name}] {item.message}'))
        return messages

    def __get_next_agent(self) -> str:
        if len(self.__history.history) == 0:
            return USER_AGENT_NAME

        messages = self.__system_prompt + self.__generate_message_history()
        messages = trim_latest_messsages(messages, 20)

        response = handle_tool_calls(self.__llm, messages, self.__llm.invoke(messages))
        response_text = remove_think_tags(response.text())

        if response_text not in self.__agent_names:
            return USER_AGENT_NAME
        return response_text
