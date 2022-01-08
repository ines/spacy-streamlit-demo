import spacy
from spacy_streamlit import visualize_ner, visualize_tokens
from spacy.matcher import Matcher
import streamlit as st

# Global setting
MODELS = {"中文(zh_core_web_sm)": "zh_core_web_sm", 
          "English(en_core_web_sm)": "en_core_web_sm", 
          "日本語(ja_core_news_sm)": "ja_core_news_sm"}
models_to_display = list(MODELS.keys())
ZH_TEXT = "（中央社）中亞國家哈薩克近日發生民眾示威暴動，引發政府切斷網路，連帶造成比特幣價格重挫，摜破4萬3000美元關卡。這也凸顯加密貨幣挖礦大國哈薩克在比特幣生態圈分量舉足輕重。"
ZH_REGEX = "\d{2,4}"
EN_TEXT = "(CNN) President Joe Biden on Thursday marked the first anniversary of the January 6 insurrection by forcefully calling out former President Donald Trump for attempting to undo American democracy, saying such an insurrection must never happen again."
EN_REGEX = "(ed|ing)"
JA_TEXT = "（朝日新聞）紙の教科書をデータ化した「デジタル教科書」が新年度から、全小中学校に無償で提供される。文部科学省が、2024年度の本格導入に向けた実証事業として外国語（英語）で配布し、希望する学校の一部には、ほかの教科からも1教科分を提供する。紙との併存や費用のあり方などについて課題を洗い出す。"
JA_REGEX = "[がでに]"
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
    # Model output
    ner_labels = nlp.get_pipe("ner").labels
    visualize_ner(doc, labels=ner_labels, show_table=False, title="命名實體")
    visualize_tokens(doc, attrs=["text", "pos_", "dep_", "ent_type_"], title="斷詞特徵")
    st.markdown("---")

with right:
    # Select num of tokens 
    selected_tok_nums = st.number_input("請選擇斷詞數量：", 1, 5, 2)
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
