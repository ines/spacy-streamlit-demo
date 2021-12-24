import spacy
from spacy_streamlit import visualize_ner, visualize_tokens
from spacy.matcher import Matcher
import streamlit as st

# Global setting
MODELS = ["zh_core_web_sm", "en_core_web_sm", "ja_core_news_sm"]
DEFAULT_TEXT = "當我正想著我到底有沒有見過孔子的時候，孔子就出現了！"
DEFAULT_REGEX = "[過了著]"
DESCRIPTION = "spaCy自然語言處理模型展示"

st.set_page_config(
    page_title=DESCRIPTION,
    page_icon="🧊",
    layout="wide",
)

# Model
selected_model = st.radio(f"{MODELS[0]}為中文模型，{MODELS[1]}為英文模型，{MODELS[2]}為日文模型", MODELS)
nlp = spacy.load(selected_model)
nlp.add_pipe("merge_entities") 
st.markdown("---")

# Text
user_text = st.text_area("請輸入文章：", DEFAULT_TEXT)
doc = nlp(user_text)
st.markdown("---")

# Two columns
left, right = st.columns(2)

with left:
    # Visualization
    ner_labels = nlp.get_pipe("ner").labels
    visualize_ner(doc, labels=ner_labels, show_table=False, title="命名實體")
    visualize_tokens(doc, attrs=["text", "pos_", "dep_", "ent_type_"], title="斷詞特徵")
    st.markdown("---")

# Pattern input
def show_one_token_attr(tok_num):
    pattern_types = ["regex", "ner"]
    selected_info = st.radio("請選擇匹配方式：", pattern_types, key="info_"+str(tok_num))
    if selected_info == pattern_types[0]:
        regex_text = st.text_input("請輸入正則表達：", DEFAULT_REGEX, key="regex_"+str(tok_num))
        pattern = [{'TEXT': {'REGEX': regex_text}}]
    elif selected_info == pattern_types[1]:
        ner_text = st.selectbox("請選擇命名實體類別：", ner_labels, key="ner_"+str(tok_num))
        pattern = [{'ENT_TYPE': ner_text}]
    return pattern 

with right:
    # Num of tokens 
    selected_tok_nums = st.number_input("請選擇斷詞數量：", 1, 5, 2)
    st.markdown("---")

    # Selected patterns
    patterns = []
    for tok_num in range(selected_tok_nums):
        pattern = show_one_token_attr(tok_num)
        patterns += pattern
    
    # Matches
    matcher = Matcher(nlp.vocab)
    matcher.add('Rule', [patterns])
    matches = matcher(doc, as_spans=True)

    # Output
    if len(matches) > 0:
        st.write("規則匹配結果：")
        for span in matches:
            text, label = span.text, span.label_
            st.write(f"{label} >>> {text}")
    else:
        st.write("沒有任何匹配結果！")
