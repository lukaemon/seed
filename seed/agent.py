from seed.memory import Conversation, Message
from seed.llm.gpt import gpt
from seed.prompt import ConversationPrompt, instruction, examples


class ConversationAgent:
    def __init__(self, agent_name: str):
        self.name = agent_name
        self.session_history = Conversation()
        self.llm = gpt

    def __call__(self, user_input: str, session_history=None) -> str:
        if session_history:  # if we have a drop in session history, use it
            self.session_history = session_history

        response = self.llm(self._build_prompt(user_input))
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
