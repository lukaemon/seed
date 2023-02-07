from typing import Optional
import logging

import discord
from discord import Message as DiscordMessage


from seed.memory import Message, Conversation
from seed.agent import ConversationAgent
from discord_bot.constant import (
    DISCORD_BOT_TOKEN,
    BOT_INVITE_URL,
    ACTIVATE_THREAD_PREFX,
    INACTIVATE_THREAD_PREFIX,
    MAX_THREAD_MESSAGES,
    AGENT_NAME,
)

from seed.llm.mt0 import MT0

logger = logging.getLogger("main.py")

logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO
)

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
tree = discord.app_commands.CommandTree(client)

logger.info("Initiating LLM backend...")
conv_agent = ConversationAgent(AGENT_NAME)
# conv_agent = ConversationAgent(
#     AGENT_NAME, llm=MT0()
# )  # uncomment to use MT0 or drop in other LLM


@client.event
async def on_ready():
    logger.info(f"We have logged in as {client.user}. Invite URL: {BOT_INVITE_URL}")
    await tree.sync()


# /chat message:
@tree.command(name="chat", description="Create a new thread for conversation")
@discord.app_commands.checks.has_permissions(send_messages=True)
@discord.app_commands.checks.has_permissions(view_channel=True)
@discord.app_commands.checks.bot_has_permissions(send_messages=True)
@discord.app_commands.checks.bot_has_permissions(view_channel=True)
@discord.app_commands.checks.bot_has_permissions(manage_threads=True)
async def chat_command(int: discord.Interaction, message: str):
    # only support creating thread in text channel
    if not isinstance(int.channel, discord.TextChannel):
        return

    user = int.user
    logger.info(f"Chat command by {user} {message[:20]}")

    embed = discord.Embed(
        description=f"<@{user.id}> wants to chat! 🤖💬",
        color=discord.Color.green(),
    )
    embed.add_field(name=user.name, value=message)

    await int.response.send_message(embed=embed)
    response = await int.original_response()

    # create the thread
    thread = await response.create_thread(
        name=f"{ACTIVATE_THREAD_PREFX} {user.name[:20]} - {message[:30]}",
        slowmode_delay=1,
        reason="gpt-bot",
        auto_archive_duration=60,
    )

    async with thread.typing():
        # init messsage, no worry about history
        response_text = conv_agent(message)
        # send the result
        await thread.send(response_text)


# calls for each message
@client.event
async def on_message(message: DiscordMessage):
    # ignore messages from the bot
    if message.author == client.user:
        return

    # ignore messages not in a thread
    channel = message.channel
    if not isinstance(channel, discord.Thread):
        return

    # ignore threads not created by the bot
    thread = channel
    if thread.owner_id != client.user.id:
        return

    # ignore threads that are archived locked or title is not what we want
    if (
        thread.archived
        or thread.locked
        or not thread.name.startswith(ACTIVATE_THREAD_PREFX)
    ):
        # ignore this thread
        return

    if thread.message_count > MAX_THREAD_MESSAGES:
        # too many messages, no longer going to reply
        await close_thread(thread=thread)
        return

    logger.info(
        f"Thread message to process - {message.author}: {message.content[:50]} - {thread.name} {thread.jump_url}"
    )

    channel_messages = [
        discord_message_to_message(message)
        async for message in thread.history(limit=MAX_THREAD_MESSAGES)
    ]
    channel_messages = [x for x in channel_messages if x is not None]
    channel_messages.reverse()

    logging.info(channel_messages)

    # take the thread history as the conversation history
    async with thread.typing():
        response_text = conv_agent(
            user_input=message.content,
            session_history=Conversation(messages=channel_messages),
        )
        # send response
        await thread.send(response_text)


async def close_thread(thread: discord.Thread):
    await thread.edit(name=INACTIVATE_THREAD_PREFIX)
    await thread.send(
        embed=discord.Embed(
            description="**Thread closed** - Context limit reached, closing...",
            color=discord.Color.blue(),
        )
    )
    await thread.edit(archived=True, locked=True)


def discord_message_to_message(message: DiscordMessage) -> Optional[Message]:
    if (
        message.type == discord.MessageType.thread_starter_message
        and message.reference.cached_message
        and len(message.reference.cached_message.embeds) > 0
        and len(message.reference.cached_message.embeds[0].fields) > 0
    ):
        field = message.reference.cached_message.embeds[0].fields[0]
        if field.value:
            return Message(author=field.name, text=field.value)
    else:
        if message.content:
            return Message(author=message.author.name, text=message.content)
    return None


client.run(DISCORD_BOT_TOKEN)
