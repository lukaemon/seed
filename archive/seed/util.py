import logging

logging.basicConfig(
    format="[%(asctime)s] [%(name)s] [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger("seed")
logger.setLevel(logging.DEBUG)
