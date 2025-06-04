from agent.agent import Agent
from agent.history import AgentHistory
from orchestrator.orchestrator import Orchestrator


def main():
    agent_history = AgentHistory("agent_history.json")
    agent_a = Agent("agent_a", "你是一个助手，你比较沉默寡言。", agent_history)
    agent_b = Agent("agent_b", "你是一个总是很冷静的人。", agent_history)
    agent_c = Agent("agent_c", "你是一个对任何事物都有好奇心的人，活泼健谈。", agent_history)

    orchestrator = Orchestrator([agent_a, agent_b, agent_c], agent_history)

    while True:
        user_input = input(">>> ")
        if user_input == "/exit":
            break

        orchestrator.chat(user_input)


if __name__ == "__main__":
    main()
