from base import Conversation, Message
from gpt import gpt
from prompt import ConversationPrompt, instruction, examples


class ConversationAgent:
    def __init__(self, agent_name: str):
        self.name = agent_name
        self.session_history = Conversation()

    def __call__(self, user_input: str) -> str:
        response = gpt(self._build_prompt(user_input))
        self._update_session_history(user_input, response)
        return response

    def _update_session_history(self, user_input: str, response: str):
        self.session_history.add_message(Message("User", user_input))
        self.session_history.add_message(Message(self.name, response))

    def _build_prompt(self, new_user_input: str):
        return ConversationPrompt(
            agent_name=self.name,
            instruction=instruction(self.name),
            examples=examples(self.name),
            session_history=self.session_history,
            new_user_input=new_user_input,
        ).render()

    def render_session_history(self):
        return self.session_history.render()


async def complete(agent_name: str, history: Conversation, new_user_input: str):
    agent = ConversationAgent(agent_name)
    agent.session_history = history

    return agent(new_user_input)
