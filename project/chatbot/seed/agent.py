from seed.memory import Conversation, Message
from seed.llm.gpt import GPT

from seed.prompt import ConversationPrompt
from seed.util import logger

gpt = GPT()  # default llm


class ConversationAgent:
    def __init__(self, agent_name: str, llm=gpt):
        """default to gpt as baseline
        :param agent_name: Name of the agent
        :param llm: text2text interface, generic to funciton and instance
        """
        self.name = agent_name
        self.session_history = Conversation()
        self.llm = llm

        logger.info(f"Initialized {self.name} agent with llm {self.llm.model_name}")

    def __call__(self, user_input: str, session_history=None) -> str:
        if session_history:  # optional drop in session history
            self.session_history = session_history

        response = self.llm(self._build_prompt(user_input))
        self._update_session_history(user_input, response)
        return response

    def _update_session_history(self, user_input: str, response: str):
        self.session_history.add_message(Message("User", user_input))
        self.session_history.add_message(Message(self.name, response))

    def _build_prompt(self, user_input: str):
        return str(
            ConversationPrompt(
                agent_name=self.name,
                session_history=self.session_history,
                user_input=user_input,
            )
        )

    def render_session_history(self):
        return self.session_history.render()
