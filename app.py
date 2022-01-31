from dragonmapper import hanzi, transcriptions
import jieba
import pandas as pd
import requests 
import spacy
from spacy_streamlit import visualize_ner, visualize_tokens
from spacy.tokens import Doc
import streamlit as st

# Global variables
MODELS = {"中文": "zh_core_web_sm", 
          "English": "en_core_web_sm", 
          "日本語": "ja_ginza"}
models_to_display = list(MODELS.keys())
ZH_TEXT = "（中央社）中央流行疫情指揮中心宣布，今天國內新增60例COVID-19（2019冠狀病毒疾病），分別為49例境外移入，11例本土病例，是去年8月29日本土新增13例以來的新高，初步研判其中10例個案皆與桃園機場疫情有關。"
MOEDICT_URL = "https://www.moedict.tw/uni/"
ZH_REGEX = "\d{2,4}"
EN_TEXT = "(CNN) Covid-19 hospitalization rates among children are soaring in the United States, with an average of 4.3 children under 5 per 100,000 hospitalized with an infection as of the week ending January 1, up from 2.6 children the previous week, according to data from the US Centers for Disease Control and Prevention. This represents a 48% increase from the week ending December 4, and the largest increase in hospitalization rate this age group has seen over the course of the pandemic."
EN_REGEX = "(ed|ing)$"
JA_TEXT = "（朝日新聞）新型コロナウイルスの国内感染者は9日、新たに8249人が確認された。2日連続で8千人を超えたのは昨年9月11日以来、約4カ月ぶり。全国的に感染拡大が進む中、年をまたいだ1週間の感染者の過半数が30代以下だった。コロナ特措法に基づく「まん延防止等重点措置」が9日から適用された3県では、広島で過去最多の619人が確認された。"
JA_REGEX = "[たい]$"
DESCRIPTION = "spaCy自然語言處理模型展示"
TOK_SEP = " | "

# Custom tokenizer class
class JiebaTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = jieba.cut(text) # returns a generator
        tokens = list(words) # convert the genetator to a list
        spaces = [False] * len(tokens)
        doc = Doc(self.vocab, words=tokens, spaces=spaces)
        return doc

st.set_page_config(
    page_icon="🤠",
    layout="wide",
)

# Choose a language model
st.markdown(f"# {DESCRIPTION}") 
st.markdown("## 語言模型") 
selected_model = st.radio("請選擇語言", models_to_display)
nlp = spacy.load(MODELS[selected_model])
          
# Merge entity spans to tokens
# nlp.add_pipe("merge_entities") 
st.markdown("---")

# Default text and regex
st.markdown("## 待分析文本") 
if selected_model == models_to_display[0]: # Chinese
    # Select a tokenizer if the Chinese model is chosen
    selected_tokenizer = st.radio("請選擇斷詞模型", ["jieba-TW", "spaCy"])
    if selected_tokenizer == "jieba-TW":
        nlp.tokenizer = JiebaTokenizer(nlp.vocab)
    default_text = ZH_TEXT
    default_regex = ZH_REGEX
elif selected_model == models_to_display[1]: # English
    default_text = EN_TEXT 
    default_regex = EN_REGEX 
elif selected_model == models_to_display[2]: # Japanese
    default_text = JA_TEXT
    default_regex = JA_REGEX 

text = st.text_area("",  default_text)
doc = nlp(text)
st.markdown("---")

# Two columns
left, right = st.columns(2)

with left:
    # Model output
    ner_labels = nlp.get_pipe("ner").labels
    visualize_ner(doc, labels=ner_labels, show_table=False, title="命名實體")
    visualize_tokens(doc, attrs=["text", "pos_", "tag_", "dep_", "head"], title="斷詞特徵")
    st.markdown("---")

with right:
    punct_and_sym = ["PUNCT", "SYM"]
    if selected_model == models_to_display[0]: # Chinese 
        st.markdown("## 原文") 
        for idx, sent in enumerate(doc.sents):
            tokens_text = [tok.text for tok in sent if tok.pos_ not in punct_and_sym]
            pinyins = [hanzi.to_pinyin(tok) for tok in tokens_text]
            display = []
            for text, pinyin in zip(tokens_text, pinyins):
                res = f"{text} [{pinyin}]"
                display.append(res)
            display_text = TOK_SEP.join(display)
            st.write(f"{idx+1} >>> {display_text}")
   
        verbs = [tok.text for tok in doc if tok.pos_ == "VERB"]
        if verbs:
            st.markdown("## 動詞")
            selected_verbs = st.multiselect("請選擇要查詢的單詞: ", verbs, verbs[0:1])
            for v in selected_verbs:
                st.write(f"### {v}")
                res = requests.get(MOEDICT_URL+v)
                if res:
                    with st.expander("點擊 + 查看更多"):
                        st.json(res.json())
                else:
                    st.write("查無結果")
            
        nouns = [tok.text for tok in doc if tok.pos_ == "NOUN"]
        if nouns:
            st.markdown("## 名詞")
            selected_nouns = st.multiselect("請選擇要查詢的單詞: ", nouns, nouns[0:1])
            for n in selected_nouns:
                st.write(f"### {n}")
                res = requests.get(MOEDICT_URL+n)
                if res:
                    with st.expander("點擊 + 查看更多"):
                        st.json(res.json())
                else:
                    st.write("查無結果")                            
                    
    elif selected_model == models_to_display[2]: # Japanese 
        st.markdown("## 原文") 
        for idx, sent in enumerate(doc.sents):
            clean_tokens = [tok for tok in sent if tok.pos_ not in punct_and_sym]
            tokens_text = [tok.text for tok in clean_tokens]
            readings = ["/".join(tok.morph.get("Reading")) for tok in clean_tokens]
            display = [f"{text} [{reading}]" for text, reading in zip(tokens_text, readings)]
            display_text = TOK_SEP.join(display)
            st.write(f"{idx+1} >>> {display_text}")          
                    
        verbs = [tok for tok in doc if tok.pos_ == "VERB"]
        if verbs:
            st.markdown("## 動詞")
            df = pd.DataFrame(
                {
                    "單詞": [tok.orth_ for tok in verbs],
                    "發音": ["/".join(tok.morph.get("Reading")) for tok in verbs],
                    "詞形變化": ["/".join(tok.morph.get("Inflection")) for tok in verbs],
                    "原形": [tok.lemma_ for tok in verbs],
                    #"正規形": [tok.norm_ for tok in verbs],
                }
            )
            st.dataframe(df)
            
        auxes = [tok for tok in doc if tok.pos_ == "AUX"]
        if auxes:
            st.markdown("## 助動詞")
            df = pd.DataFrame(
                {
                    "單詞": [tok.orth_ for tok in auxes],
                    "發音": ["/".join(tok.morph.get("Reading")) for tok in auxes],
                    "詞形變化": ["/".join(tok.morph.get("Inflection")) for tok in auxes],
                    "原形": [tok.lemma_ for tok in auxes],
                    #"正規形": [tok.norm_ for tok in auxes],
                }
            )
            st.dataframe(df)

    else:
        st.write("Work in progress")
