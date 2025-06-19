import json
import logging
from typing import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START
from agent.history import AgentHistory
from tools.tools import user_dict, UserData, tools
from agent.prompts import system_prompt, system_prompt_decide_reply
from utils.utils import handle_tool_calls, remove_think_tags, trim_latest_messsages
from utils.constants import *


class AgentState(TypedDict):
    history: AgentHistory
    should_reply: bool


STATE_DECIDE_REPLY = "decide_reply"
STATE_GENERATE_RESPONSE = "generate_response"

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fileLogger = logging.FileHandler("agent_activities.log")
fileLogger.setLevel(logging.DEBUG)
logFormatter = logging.Formatter('%(asctime)s|%(name)s|%(levelname)s|%(message)s')
fileLogger.setFormatter(logFormatter)
logger.addHandler(fileLogger)


class Agent:
    def __init__(self, name: str, instructions: str, history: AgentHistory):
        self.__name = name
        self.__history = history

        self.__system_prompt = ChatPromptTemplate([('system', system_prompt)]).format_messages(name=name, instructions=instructions)
        self.__system_prompt_decide_reply = ChatPromptTemplate(
            [('system', system_prompt_decide_reply)]).format_messages(name=name, instructions=instructions)

        user_dict[name] = UserData(instructions=instructions)

        self.__llm = ChatOllama(base_url=OLLAMA_URL, model=OLLAMA_MODEL).bind_tools(tools)

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node(STATE_DECIDE_REPLY, self.__decide_reply)
        graph_builder.add_node(STATE_GENERATE_RESPONSE, self.__generate_response)  # TODO
        graph_builder.add_edge(START, STATE_DECIDE_REPLY)
        graph_builder.add_edge(STATE_DECIDE_REPLY, STATE_GENERATE_RESPONSE)
        self.__workflow = graph_builder.compile()

        self.__logger = logging.getLogger(__name__)

    @property
    def name(self) -> str:
        return self.__name

    def run(self):
        self.__workflow.invoke(AgentState(history=self.__history, should_reply=False))

    def __decide_reply(self, state: AgentState) -> AgentState:
        history = self.__system_prompt_decide_reply + state['history'].get_llm_messages_based_on(self.__name)
        history = trim_latest_messsages(history, 20)
        response = handle_tool_calls(self.__llm, history, self.__llm.invoke(history))

        try:
            response_json = json.loads(response.text())
        except json.JSONDecodeError:
            response_json = {'should_reply': False, 'reason': f'Error: invalid JSON\n{response.text()}'}
            self.__logger.warning(f'Agent "{self.__name}" decides not to reply due to invalid JSON: {response.text()}')

        if response_json['should_reply']:
            state['should_reply'] = True
            self.__logger.info(f'Agent "{self.__name}" decides to reply because: {response_json["reason"]}')
        else:
            state['should_reply'] = False
            self.__logger.info(f'Agent "{self.__name}" decides not to reply because: {response_json["reason"]}')

        return state

    def __generate_response(self, state: AgentState) -> AgentState:
        history = self.__system_prompt + state['history'].get_llm_messages_based_on(self.__name)
        history = trim_latest_messsages(history, 20)
        response = handle_tool_calls(self.__llm, history, self.__llm.invoke(history))

        response_text = remove_think_tags(response.text())
        if response_text:
            state['history'].add(self.__name, remove_think_tags(response.text()))

        return state
