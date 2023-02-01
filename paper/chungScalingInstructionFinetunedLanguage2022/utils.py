import logging

logging.basicConfig(
    format="[%(levelname)s] [%(asctime)s] [%(name)s] [%(filename)s:%(lineno)d] %(message)s",
)

logger = logging.getLogger("flan-t5")
logger.setLevel(logging.INFO)
