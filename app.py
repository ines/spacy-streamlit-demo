import spacy_streamlit
from pathlib import Path
import srsly
import importlib
import random

MODELS = srsly.read_json(Path(__file__).parent / "models.json")
DEFAULT_MODEL = "en_core_web_sm"
DEFAULT_TEXT = "David Bowie moved to the US in 1974, initially staying in New York City before settling in Los Angeles."
DESCRIPTION = """**Explore trained [spaCy v3.0](https://nightly.spacy.io) pipelines**"""

def get_default_text(nlp):
    # Check if spaCy has built-in example texts for the language
    try:
        examples = importlib.import_module(f".lang.{nlp.lang}.examples", "spacy")
        return examples.sentences[0]
    except (ModuleNotFoundError, ImportError):
        return ""

spacy_streamlit.visualize(
    MODELS,
    default_model=DEFAULT_MODEL,
    visualizers=["parser", "ner", "similarity", "tokens"],
    show_visualizer_select=True,
    sidebar_description=DESCRIPTION,
    get_default_text=get_default_text
)
