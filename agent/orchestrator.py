from typing import Iterable

from agent.agent import Agent
from agent.history import AgentHistory


USER_AGENT_NAME = "user"


class AgentOrchestrator:
    def __init__(self, agents: Iterable[Agent], history: AgentHistory):
        self.__agents = list(agents)
        self.__history = history

    def chat(self, message: str):
        self.__history.add(USER_AGENT_NAME, message)

        while True:
            history_length = len(self.__history.history)
            for agent in self.__agents:
                agent.receive_message()
            if len(self.__history.history) == history_length:
                # No new messages, stop
                break
