import json
from typing import Callable
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel


class AgentHistoryItem(BaseModel):
    name: str
    message: str


class AgentHistory:
    def __init__(self, file_path: str):
        self.__file_path = file_path
        self.__history: list[AgentHistoryItem] = []

    @property
    def history(self):
        return self.__history

    def add(self, name: str, message: str):
        self.__history.append(AgentHistoryItem(name=name, message=message))

    def get_llm_messages_based_on(self, name: str) -> list[BaseMessage]:
        messages: list[BaseMessage] = []

        for item in self.__history:
            if item.name == name:
                messages.append(AIMessage(content=item.message))
            else:
                if len(messages) == 0 or isinstance(messages[-1], AIMessage):
                    messages.append(HumanMessage(content=f'[{item.name}] {item.message}'))
                else:
                    messages[-1].content += f'\n[{item.name}] {item.message}'

        return messages

    def dump_json(self):
        with open(self.__file_path, 'w', encoding='utf-8') as fp:
            json.dump([i.model_dump() for i in self.__history], fp=fp, ensure_ascii=False, indent=2)

    def load_json(self):
        with open(self.__file_path, 'r', encoding='utf-8') as fp:
            self.__history = [AgentHistoryItem(**i) for i in json.load(fp)]
