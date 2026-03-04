"""Constants for the Knowledge executor.

Enums and defaults for knowledge proposition management.
"""

from enum import Enum


class LifecycleState(str, Enum):
    """Proposition lifecycle states matching knowledge_propositions.lifecycle_state."""

    ACTIVE = "ACTIVE"
    QUARANTINED = "QUARANTINED"
    FLAGGED = "FLAGGED"
    ARCHIVED = "ARCHIVED"


class Authority(str, Enum):
    """Proposition authority levels matching knowledge_propositions.authority."""

    EXTRACTED = "EXTRACTED"
    COMMUNITY_SUMMARY = "COMMUNITY_SUMMARY"
    USER_VALIDATED = "USER_VALIDATED"
    AGENT = "AGENT"


# Search defaults
DEFAULT_LIMIT = 10
DEFAULT_MIN_CONFIDENCE = 0.3
DEFAULT_VECTOR_WEIGHT = 0.6
DEFAULT_FTS_WEIGHT = 0.4
DEFAULT_RRF_K = 60

# Context defaults
DEFAULT_MAX_TOKENS = 4000
DEFAULT_MMR_LAMBDA = 0.7

# Token estimation: ~4 chars per token for English text
CHARS_PER_TOKEN = 4
