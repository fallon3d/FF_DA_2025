# Package metadata
__version__ = "0.1.0"

# Re-export common entry points for convenience
from .core.evaluation import evaluate_players, SCORING_DEFAULT
from .core.suggestions import suggest

__all__ = ["evaluate_players", "SCORING_DEFAULT", "suggest", "__version__"]
