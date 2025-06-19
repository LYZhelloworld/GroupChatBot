import json
import re

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, trim_messages
from langchain_core.runnables.base import Runnable
from tools.tools import tools


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


def handle_tool_calls(llm: Runnable[LanguageModelInput, BaseMessage], history: list[BaseMessage], response: BaseMessage):
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
                    tmp_history.append(ToolMessage(content=json.dumps(tool_response), tool_call_id=tool_id))

        response = llm.invoke(tmp_history)

    return response


def trim_latest_messsages(messages: list[BaseMessage], n: int):
    return trim_messages(
        messages,
        strategy='last',
        token_counter=len,
        max_tokens=n,
        start_on='human',
        end_on=('human', 'tool'),
        include_system=True,
        allow_partial=False)
