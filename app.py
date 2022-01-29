import spacy
from spacy_streamlit import visualize_ner, visualize_tokens
from spacy.matcher import Matcher
import streamlit as st
import jieba

# Global setting
MODELS = {"中文(zh_core_web_sm)": "zh_core_web_sm", 
          "English(en_core_web_sm)": "en_core_web_sm", 
          "日本語(ja_core_news_sm)": "ja_core_news_sm"}
models_to_display = list(MODELS.keys())
ZH_TEXT = "（中央社）中央流行疫情指揮中心宣布，今天國內新增60例COVID-19（2019冠狀病毒疾病），分別為49例境外移入，11例本土病例，是去年8月29日本土新增13例以來的新高，初步研判其中10例個案皆與桃園機場疫情有關。"
ZH_REGEX = "\d{2,4}"
EN_TEXT = "(CNN) Covid-19 hospitalization rates among children are soaring in the United States, with an average of 4.3 children under 5 per 100,000 hospitalized with an infection as of the week ending January 1, up from 2.6 children the previous week, according to data from the US Centers for Disease Control and Prevention. This represents a 48% increase from the week ending December 4, and the largest increase in hospitalization rate this age group has seen over the course of the pandemic."
EN_REGEX = "(ed|ing)$"
JA_TEXT = "（朝日新聞）新型コロナウイルスの国内感染者は9日、新たに8249人が確認された。2日連続で8千人を超えたのは昨年9月11日以来、約4カ月ぶり。全国的に感染拡大が進む中、年をまたいだ1週間の感染者の過半数が30代以下だった。コロナ特措法に基づく「まん延防止等重点措置」が9日から適用された3県では、広島で過去最多の619人が確認された。"
JA_REGEX = "[たい]$"
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
    default_text = EN_TEXT 
    default_regex = EN_REGEX 
elif selected_model == models_to_display[2]:
    default_text = JA_TEXT
    default_regex = JA_REGEX 

user_text = st.text_area("請輸入文章：", default_text)
jieba_res = jieba.cut(user_text) 
jieba_res = "|".join(jieba_res)
st.write(f"Jieba: {jieba_res}")

doc = nlp(user_text)
spacy_res = [tok.text for tok in doc]
spacy_res = "|".join(spacy_res)
st.write(f"spaCy: {spacy_res}")
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
    # Model output
    ner_labels = nlp.get_pipe("ner").labels
    visualize_ner(doc, labels=ner_labels, show_table=False, title="命名實體")
    visualize_tokens(doc, attrs=["text", "pos_", "dep_", "ent_type_"], title="斷詞特徵")
    st.markdown("---")

with right:
    # Select num of tokens 
    selected_tok_nums = st.number_input("請選擇斷詞數量：", 1, 5, 1)
    st.markdown("---")

    # Select patterns
    patterns = []
    for tok_num in range(selected_tok_nums):
        pattern = show_one_token_attr(tok_num)
        patterns += pattern
    
    # Match the text with the selected patterns
    matcher = Matcher(nlp.vocab)
    matcher.add('Rule', [patterns])
    matches = matcher(doc, as_spans=True)

    # Output
    if matches:
        st.markdown("## 規則匹配結果：")
        for span in matches:
            text = span.text
            left_toks = span.lefts
            left_texts = [t.text for t in left_toks]
            right_toks = span.rights
            right_texts = [t.text for t in right_toks]
            st.write(f"{left_texts} **{text}** {right_texts}")
    else:
        st.markdown("## 沒有任何匹配結果！")
