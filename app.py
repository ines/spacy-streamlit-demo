import spacy_streamlit
from pathlib import Path
import srsly

MODELS = srsly.read_json(Path(__file__).parent / "models.json")
DEFAULT_TEXT = "Sundar Pichai is the CEO of Google."
DESCRIPTION = """**Explore trained [spaCy v3.0](https://nightly.spacy.io) pipelines**"""

spacy_streamlit.visualize(
    MODELS,
    DEFAULT_TEXT,
    visualizers=["parser", "ner", "similarity", "tokens"],
    show_visualizer_select=True,
    # TODO: enable when morph issue is fixed in next nightly
    show_json_doc=False,
    sidebar_description=DESCRIPTION
)
