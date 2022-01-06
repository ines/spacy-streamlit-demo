import spacy
from spacy_streamlit import visualize_ner, visualize_tokens
from spacy.matcher import Matcher
import streamlit as st

# Global setting
MODELS = {"中文": "zh_core_web_sm", 
          "English": "en_core_web_sm", 
          "日本語": "ja_core_news_sm"}
models_to_display = list(MODELS.keys())
ZH_TEXT = "當我正想著我到底有沒有見過孔子的時候，孔子就出現了！"
ZH_REGEX = "[過了著]"
DESCRIPTION = "spaCy自然語言處理模型展示"

st.set_page_config(
    page_icon="🤠",
    layout="wide",
)

# Model
st.markdown(f"# {DESCRIPTION}") 
st.markdown("## 語言模型") 
selected_model = st.radio("請選擇語言", models_to_display)
nlp = spacy.load(MODELS[selected_model])
nlp.add_pipe("merge_entities") 
st.markdown("---")

# Text
st.markdown("## 待分析文本") 
if selected_model == models_to_display[0]:
    default_text = ZH_TEXT
    default_regex = ZH_REGEX
elif selected_model == models_to_display[1]:
    default_text = ZH_TEXT # to be replaced
    default_regex = ZH_REGEX # to be replaced
elif selected_model == models_to_display[2]:
    default_text = ZH_TEXT # to be replaced
    default_regex = ZH_REGEX # to be replaced 

user_text = st.text_area("請輸入文章：", default_text)
doc = nlp(user_text)
st.markdown("---")

# Pattern input
def show_one_token_attr(tok_num):
    pattern_types = ["正則表達", "命名實體"]
    selected_info = st.radio("請選擇匹配方式：", pattern_types, key="info_"+str(tok_num))
    if selected_info == pattern_types[0]:
        regex_text = st.text_input("請輸入正則表達：", default_regex, key="regex_"+str(tok_num))
        pattern = [{'TEXT': {'REGEX': regex_text}}]
    elif selected_info == pattern_types[1]:
        ner_text = st.selectbox("請選擇命名實體類別：", ner_labels, key="ner_"+str(tok_num))
        pattern = [{'ENT_TYPE': ner_text}]
    return pattern 

# Two columns
left, right = st.columns(2)

with left:
    # Visualization
    ner_labels = nlp.get_pipe("ner").labels
    visualize_ner(doc, labels=ner_labels, show_table=False, title="命名實體")
    visualize_tokens(doc, attrs=["text", "pos_", "dep_", "ent_type_"], title="斷詞特徵")
    st.markdown("---")

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
    if matches:
        st.markdown("## 規則匹配結果：")
        for span in matches:
            text = span.text
            #left_tokens = span.lefts
            #left_chunks = [t.txt for t in left_tokens]
            #right_tokens = span.rights
            #right_chunks = [t.txt for t in right_tokens]
            st.markdown(f"### {text}")
    else:
        st.markdownn("## 沒有任何匹配結果！")
