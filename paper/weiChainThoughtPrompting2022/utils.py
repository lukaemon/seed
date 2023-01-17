import logging


logging.basicConfig(
    format="[%(asctime)s] [%(name)s] [%(filename)s:%(lineno)d] %(message)s",
)

logger = logging.getLogger("CoT")
logger.setLevel(logging.INFO)
