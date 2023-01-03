import yaml
import os
from dotenv import load_dotenv

load_dotenv()

# load .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
HUGGINFACEHUB_API_TOKEN = os.getenv("HUGGINFACEHUB_API_TOKEN")
DISCORD_CLIENT_ID = os.getenv("DISCORD_CLIENT_ID")
ALLOWED_SERVER_IDS = os.getenv("ALLOWED_SERVER_IDS")

# load config.yaml
CONFIG_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_YAML_PATH = os.path.join(CONFIG_DIR, "config.yaml")
with open(CONFIG_YAML_PATH, "r") as f:
    config = yaml.safe_load(f)


AGENT_NAME = "CoCo"
BOT_INVITE_URL = f"https://discord.com/api/oauth2/authorize?client_id={DISCORD_CLIENT_ID}&permissions=328565073920&scope=bot"
MAX_THREAD_MESSAGES = 200
ACTIVATE_THREAD_PREFX = "💬✅"
INACTIVATE_THREAD_PREFIX = "💬❌"
MAX_CHARS_PER_REPLY_MSG = (
    1500  # discord has a 2k limit, we just break message into 1.5k
)
