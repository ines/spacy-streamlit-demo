from dragonmapper import hanzi, transcriptions
import jieba
import pandas as pd
import re
import requests 
import spacy
from spacy_streamlit import visualize_ner, visualize_tokens
#from spacy.language import Language
from spacy.tokens import Doc
import streamlit as st

# Global variables
MODELS = {"中文": "zh_core_web_sm", 
          "English": "en_core_web_sm", 
          "日本語": "ja_ginza"}
models_to_display = list(MODELS.keys())
ZH_TEXT = """（中央社）迎接虎年到來，台北101今天表示，即日起推出「虎年新春燈光秀」，將持續至2月5日，每晚6時至10時，除整點會有報時燈光變化外，每15分鐘還會有3分鐘的燈光秀。台北101下午透過新聞稿表示，今年特別設計「虎年新春燈光秀」，從今晚開始閃耀台北天際線，一直延續至2月5日，共7天。"""
ZH_REGEX = "\d{2,4}[\u4E00-\u9FFF]+"
EN_TEXT = """(CNN) Not all Lunar New Year foods are created equal. Some only make a brief appearance at the festival for auspicious purposes. Others are so delicious they grace dim sum tables around the world all year.
Turnip cake -- called "loh bak goh" in Cantonese -- falls into the latter category.
Chef Tsang Chiu King, culinary director of Ming Court in Hong Kong's Wan Chai area, has his own theory on why turnip cake is such a popular Lunar New Year dish, especially in southern China.
"Compared to other Lunar New Year cakes, turnip cake is popular as it's one of the few savory new year puddings. Together with the freshness of the white radish, it can be quite addictive as a snack or a main dish," he says."""
EN_REGEX = "(ed|ing)$"
JA_TEXT = """（朝日新聞）寅（とら）年の2022年を前に、90種480匹の野生動物を飼育する「到津（いとうづ）の森公園」（北九州市）が盛り上がっている。同園のマスコットはアムールトラのミライ（雌、10歳）。22年は「ニャーニャー」の年としてネコ好きの間で話題となっており、「干支（えと）で唯一のネコ科のトラ人気につながれば」と期待している。"""
JA_REGEX = "[たい]$"
DESCRIPTION = "AI模型輔助語言學習"
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
    
# Utility functions
def create_jap_df(tokens):
    seen_texts = []
    filtered_tokens = []
    for tok in tokens:
        if tok.text not in seen_texts:
            filtered_tokens.append(tok)
            
    df = pd.DataFrame(
      {
          "單詞": [tok.orth_ for tok in filtered_tokens],
          "發音": ["/".join(tok.morph.get("Reading")) for tok in filtered_tokens],
          "詞形變化": ["/".join(tok.morph.get("Inflection")) for tok in filtered_tokens],
          "原形": [tok.lemma_ for tok in filtered_tokens],
          #"正規形": [tok.norm_ for tok in verbs],
      }
    )
    st.dataframe(df)
    csv = df.to_csv().encode('utf-8')
    st.download_button(
      label="下載表格",
      data=csv,
      file_name='jap_forms.csv',
      )

def filter_tokens(doc):
    clean_tokens = [tok for tok in doc if tok.pos_ not in ["PUNCT", "SYM"]]
    clean_tokens = [tok for tok in clean_tokens if not tok.like_email]
    clean_tokens = [tok for tok in clean_tokens if not tok.like_url]
    clean_tokens = [tok for tok in clean_tokens if not tok.like_num]
    clean_tokens = [tok for tok in clean_tokens if not tok.is_punct]
    clean_tokens = [tok for tok in clean_tokens if not tok.is_space]
    return clean_tokens
            
def moedict_caller(word):
    st.write(f"### {word}")
    req = requests.get(f"https://www.moedict.tw/a/{word}.json")
    if req:
        with st.expander("點擊 + 檢視結果"):
            st.json(req.json())
    else:
        st.write("查無結果")
          
# Page setting
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
    selected_tokenizer = st.radio("請選擇斷詞模型", ["spaCy", "jieba-TW"])
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

st.info("修改文本後，按下Ctrl + Enter以更新")
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
        st.markdown("## 分析後文本") 
        for idx, sent in enumerate(doc.sents):
            tokens_text = [tok.text for tok in sent if tok.pos_ not in punct_and_sym]
            pinyins = [hanzi.to_pinyin(word) for word in tokens_text]
            display = []
            for text, pinyin in zip(tokens_text, pinyins):
                res = f"{text} [{pinyin}]"
                display.append(res)
            display_text = TOK_SEP.join(display)
            st.write(f"{idx+1} >>> {display_text}")
        
        st.markdown("## 單詞解釋")
        clean_tokens = filter_tokens(doc)
        alphanum_pattern = re.compile(r"[a-zA-Z0-9]")
        clean_tokens_text = [tok.text for tok in clean_tokens if not alphanum_pattern.search(tok.text)]
        vocab = list(set(clean_tokens_text))
        if vocab:
            selected_words = st.multiselect("請選擇要查詢的單詞: ", vocab, vocab[0:3])
            for w in selected_words:
                moedict_caller(w)                        
                    
    elif selected_model == models_to_display[2]: # Japanese 
        st.markdown("## 分析後文本") 
        for idx, sent in enumerate(doc.sents):
            clean_tokens = [tok for tok in sent if tok.pos_ not in ["PUNCT", "SYM"]]
            tokens_text = [tok.text for tok in clean_tokens]
            readings = ["/".join(tok.morph.get("Reading")) for tok in clean_tokens]
            display = [f"{text} [{reading}]" for text, reading in zip(tokens_text, readings)]
            display_text = TOK_SEP.join(display)
            st.write(f"{idx+1} >>> {display_text}")          
        
        st.markdown("## 詞形變化")
        # Collect inflected forms
        inflected_forms = [tok for tok in doc if tok.tag_.startswith("動詞") or tok.tag_.startswith("形")]
        if inflected_forms:
            create_jap_df(inflected_forms)

    else:
        st.write("Work in progress")
