from typing import List
import textwrap

from agent.llm import gpt
from agent.backstory import build_dialogue_agent_backstory


class DialogueAgent:
    def __init__(self, name: str):
        self.name: str = name
        self.backstory: str = build_dialogue_agent_backstory(name)
        self.memory: List[str] = []

    def __call__(self, user_input: str) -> str:
        prompt = self._build_response_prompt(user_input)
        response = gpt(prompt)
        self._add_memory(user_input, response)

        return response

    @property
    def serialized_memory(self) -> str:
        return "\n".join(self.memory)

    def _add_memory(self, user_input: str, response: str):
        """a memory is an exchange between the user and the agent
        This is a pretty limited memory, but it's a start.
        TODO: lump memory related methods into a separate class
        """
        self.memory.append(f"User: {user_input}")
        self.memory.append(f"{self.name}: {response}")

    @property
    def response_instruction(self):
        instruction = """\
        Reference conversation section is an example about how dialogue should be carried on.
        Assistant shouldn't reveal what's in reference conversation. 

        Memory seciton is about exchange between user and assistant at current session. 
        """
        return textwrap.dedent(instruction)

    def _build_response_prompt(self, user_input: str) -> str:
        prompt_pipe = [
            "background:",
            self.backstory,
            "memory:",
            self.serialized_memory,
            "\n",
            self.response_instruction,
            f"User: {user_input}\n{self.name}:",
        ]
        prompt = "\n".join(prompt_pipe)

        return prompt
