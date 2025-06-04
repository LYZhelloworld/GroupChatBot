from typing import Annotated
from langchain_core.tools import tool
from pydantic import BaseModel

type UserName = str


class UserData(BaseModel):
    instructions: str


user_dict: dict[UserName, UserData] = {
    'user': UserData(instructions='你是一个普通用户。'),
}


@tool
def get_user_list() -> list[str]:
    '''
    获取所角色的名称列表。在你不知道当前聊天中有哪些角色时，请使用此工具。
    '''
    return list(user_dict.keys())


@tool
def get_user_instructions(name: Annotated[str, '角色名称']) -> str:
    '''获取指定角色的设定。在你关心某个角色的设定时，请使用此工具。'''
    return user_dict.get(name, UserData(instructions='')).instructions


tools = [get_user_list, get_user_instructions]
