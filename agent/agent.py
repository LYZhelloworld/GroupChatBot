import re

from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.messages.tool import ToolMessage
from langchain_ollama import ChatOllama
from agent.history import AgentHistory
from agent.tools import user_dict, UserData, tools


class Agent:
    def __init__(self, name: str, instructions: str, history: AgentHistory):
        self.__name = name
        self.__history = history

        user_dict[name] = UserData(instructions=instructions)

        llm = ChatOllama(model="qwen3:8b", base_url='http://localhost:11434')
        self.__llm = llm.bind_tools(tools)

    def receive_message(self):
        history = self.__history.get_llm_messages_based_on(self.__name)
        response = self.__llm.invoke(history)

        response = self.__handle_tool_calls(history, response)

        response_text = remove_think_tags(response.text())
        if response_text:
            self.__history.add(self.__name, remove_think_tags(response.text()))

    def __handle_tool_calls(self, history: list[BaseMessage], response: BaseMessage) -> BaseMessage:
        tmp_history = history.copy()

        while isinstance(response, AIMessage) and len(response.tool_calls) > 0:
            tmp_history.append(response)
            # Handle tool calls
            for tool in response.tool_calls:
                tool_id = tool['id']
                tool_name = tool['name']
                tool_args = tool['args']
                for t in tools:
                    if t.name == tool_name:
                        tool_response = t.invoke(tool_args)
                        tmp_history.append(ToolMessage(content=tool_response, tool_call_id=tool_id))

            response = self.__llm.invoke(history + tmp_history)

        return response


def remove_think_tags(text: str) -> str:
    """
    Removes content enclosed within `<think>` and `</think>` tags from the given text.

    - If both opening and closing `<think>` tags are present, the content within the tags is removed.
    - If only the opening `<think>` tag is present without a closing tag, an empty string is returned.
    - If no `<think>` tags are present, the text is returned as-is after trimming whitespace.

    :param str text: The input text to process.
    :return: The processed text with `<think>` tags handled appropriately.
    :rtype: str
    """

    if "<think>" in text and "</think>" in text:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    elif "<think>" in text:
        # The tag is not closed.
        return ""
    else:
        return text.strip()
