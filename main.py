from agent.agent import Agent
from agent.history import AgentHistory
from agent.orchestrator import AgentOrchestrator


def main():
    agent_history = AgentHistory("agent_history.json")
    agent_a = Agent("Agent A", "You are Agent A", agent_history)
    agent_b = Agent("Agent B", "You are Agent B", agent_history)
    agent_c = Agent("Agent C", "You are Agent C", agent_history)

    orchestrator = AgentOrchestrator([agent_a, agent_b, agent_c], agent_history)
    orchestrator.chat_message_hook = lambda name, message: print(f"{name}: {message}")

    while True:
        user_input = input(">>> ")
        if user_input == "/exit":
            break

        orchestrator.chat(user_input)


if __name__ == "__main__":
    main()
