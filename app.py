import spacy_streamlit
from pathlib import Path
import srsly

MODELS = srsly.read_json(Path(__file__).parent / "models.json")
DEFAULT_TEXT = "Sundar Pichai is the CEO of Google."

spacy_streamlit.visualize(
    MODELS,
    DEFAULT_TEXT,
    visualizers=["parser", "ner", "similarity", "tokens"],
    show_visualizer_select=True,
)
