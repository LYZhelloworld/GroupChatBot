from ollama import Client
from prompt_toolkit import prompt
from agent.agent import Agent
from agent.history import AgentHistory
from utils.constants import *


def prepare_model():
    client = Client(host=OLLAMA_URL)
    list_response = client.list()
    if all(OLLAMA_MODEL != model.model for model in list_response.models):
        print(f"Model {OLLAMA_MODEL} not found. Pulling from Ollama...")
        pull_response = client.pull(OLLAMA_MODEL, stream=True)
        for delta in pull_response:
            if (delta.completed is not None and delta.total is not None):
                print(f'{delta.status}: {delta.completed}/{delta.total}')
            else:
                print(delta.status)
    print(f'Loading model {OLLAMA_MODEL}...')
    client.generate(model=OLLAMA_MODEL)
    print('Loaded.')


def main():
    prepare_model()

    agent_history = AgentHistory("agent_history.json")
    agent_a = Agent("agent_a", "你是一个助手，你比较沉默寡言。", agent_history)
    agent_b = Agent("agent_b", "你是一个总是很冷静的人。", agent_history)
    agent_c = Agent("agent_c", "你是一个对任何事物都有好奇心的人，活泼健谈。", agent_history)

    agents = [agent_a, agent_b, agent_c]
    agent_history.print_all_history()

    while True:
        user_input = input(">>> ").strip()
        if user_input == "/exit":
            break

        if user_input:
            agent_history.add(USER_AGENT_NAME, user_input)

        for agent in agents:
            agent.run()


if __name__ == "__main__":
    main()
