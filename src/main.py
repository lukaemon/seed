import logging

import discord
from discord import Message as DiscordMessage

from base import Message, Conversation
from agent import complete
from config.constant import (
    DISCORD_BOT_TOKEN,
    DISCORD_CLIENT_ID,
    ALLOWED_SERVER_IDS,
    BOT_INVITE_URL,
    ACTIVATE_THREAD_PREFX,
    MAX_THREAD_MESSAGES,
    AGENT_NAME,
)
from utils import close_thread, discord_message_to_message

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO
)

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
tree = discord.app_commands.CommandTree(client)


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
        # fetch completion
        response_text = await complete(AGENT_NAME, Conversation(), message)
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

    # generate the response
    async with thread.typing():
        response_text = await complete(
            AGENT_NAME,
            history=Conversation(messages=channel_messages),
            new_user_input=message.content,
        )

    # send response
    await thread.send(response_text)


client.run(DISCORD_BOT_TOKEN)
