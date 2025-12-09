"""Logging utilities for the RAG system."""

import sys
from pathlib import Path
from loguru import logger
from utils.config import config

# Remove default handler
logger.remove()

# Add console handler
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level=config.log_level,
    colorize=True
)

# Add file handler
log_path = Path(config.log_file)
log_path.parent.mkdir(parents=True, exist_ok=True)

logger.add(
    config.log_file,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
    level=config.log_level,
    rotation="10 MB",
    retention="30 days",
    compression="zip"
)

__all__ = ["logger"]
