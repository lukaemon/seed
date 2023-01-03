import yaml
import os
from dotenv import load_dotenv

load_dotenv()

# load .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
HUGGINFACEHUB_API_TOKEN = os.getenv("HUGGINFACEHUB_API_TOKEN")

# load config.yaml
CONFIG_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_YAML_PATH = os.path.join(CONFIG_DIR, "config.yaml")
with open(CONFIG_YAML_PATH, "r") as f:
    config = yaml.safe_load(f)

AGENT_NAME = config["agent_name"]
