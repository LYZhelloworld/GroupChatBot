from langchain_core.tools import tool
from pydantic import BaseModel

type UserName = str


class UserData(BaseModel):
    instructions: str


user_dict: dict[UserName, UserData] = {}


@tool
def get_user_list() -> list[str]:
    '''获取所角色的名称列表'''
    return list(user_dict.keys())


@tool
def get_user_instructions(name: str) -> str:
    '''获取指定角色的设定'''
    return user_dict.get(name, UserData(instructions='')).instructions


tools = [get_user_list, get_user_instructions]
