import spacy_streamlit
from pathlib import Path
import srsly

MODELS = srsly.read_json(Path(__file__).parent / "models.json")
DEFAULT_MODEL = "en_core_web_sm"
DEFAULT_TEXT = "Sundar Pichai is the CEO of Google."
DESCRIPTION = """**Explore trained [spaCy v3.0](https://nightly.spacy.io) pipelines**"""

spacy_streamlit.visualize(
    MODELS,
    DEFAULT_TEXT,
    default_model=DEFAULT_MODEL,
    visualizers=["parser", "ner", "similarity", "tokens"],
    show_visualizer_select=True,
    sidebar_description=DESCRIPTION
)
