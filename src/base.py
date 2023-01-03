from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class Message:
    author: str
    text: Optional[str] = None

    def render(self):
        return f"{self.author}: {self.text if self.text else ''}"


@dataclass
class Conversation:
    messages: List[Message] = field(default_factory=list)

    def render(self):
        return "\n".join([m.render() for m in self.messages])

    def add_message(self, message: Message):
        self.messages.append(message)
