from typing import Callable, Iterable

from agent.agent import Agent
from agent.history import AgentHistory


USER_AGENT_NAME = "user"


class AgentOrchestrator:
    def __init__(self, agents: Iterable[Agent], history: AgentHistory):
        self.__agents = list(agents)
        self.__history = history
        self.__chat_message_hook: Callable[[str, str], None] = lambda name, message: None

    @property
    def chat_message_hook(self) -> Callable[[str, str], None]:
        return self.__chat_message_hook

    @chat_message_hook.setter
    def chat_message_hook(self, hook: Callable[[str, str], None]):
        self.__chat_message_hook = hook

    def chat(self, message: str):
        self.__history.add(USER_AGENT_NAME, message)

        while True:
            history_length = len(self.__history.history)
            for agent in self.__agents:
                agent.receive_message()
                if len(self.__history.history) > history_length:
                    last_message = self.__history.history[-1]
                    self.__chat_message_hook(last_message.name, last_message.message)
            if len(self.__history.history) == history_length:
                # No new messages, stop
                break
