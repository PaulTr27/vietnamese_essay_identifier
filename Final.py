from vncorenlp import VnCoreNLP
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
from annotated_text import annotated_text, annotation
import os
import re
import textdistance as td
import pandas as pd
import joblib
import string
import streamlit as st
import numpy as np
import spellcor as sc
from spellcor import spellcorrect
@st.cache(allow_output_mutation=True,show_spinner=False)
def load_model(model_path,tokenizer_path,seg_path):
    segmenter = VnCoreNLP(seg_path, annotators="wseg", max_heap_size='-Xmx500m') 
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer, segmenter
    # In[6]:
def introduction():
    st.title('This is the new title of my introduction page')


def application(): 
    st.write("Welcome to your friendly app")
    cwd = os.getcwd() 
    segmenter_path = os.path.join(cwd,'VnCoreNLP-1.1.1.jar')
    with st.spinner(text="Initializing..."):
        classifier, tokenizer, segmenter = load_model("PaulTran/vietnamese_essay_identify",
                                                      "vinai/phobert-base",
                                                      segmenter_path)
    essay = st.text_area('Write your essay here', "Đây là ví dụ",height=600,key = 'input_text')
    sentences_list = essay.split('.')

    submission = st.button('Submit')
    if submission:
        pred_labels_list = []
        misspelled = []
        label_names = ['Nghị Luận','Biểu cảm','Miêu tả','Tự sự', 'Thuyết minh']
        for sentence in sentences_list:
            sentence = re.sub(r'[^\w\s]', '', sentence.lower())
            segment = segmenter.tokenize(sentence)
            if len(segment) > 0:
                for word in segment[0]:
                    
                    if word not in segmented_vocab and word not in decode_dict.keys():
                        misspelled.append(word)
            tokenized_txt = tokenizer(sentence,
                                      max_length=tokenizer.model_max_length,
                                      truncation=True,
                                      padding=True,
                                      return_tensors="pt"
                                     )
            input_ids, mask = tokenized_txt['input_ids'], tokenized_txt['attention_mask']
            if input_ids is not None:
                with torch.no_grad():
                    logits = classifier(input_ids, mask).logits
                prediction = torch.argmax(logits, dim=1)
                pred_labels_list.append(prediction)
            
        pred_prob = []
        for i in range(4):
            pred_prob.append((pred_labels_list.count(i)/len(pred_labels_list)))
        pred_label_name = label_names[np.argmax(pred_prob)]
        st.write('Your essay is a {} type'.format(pred_label_name))
        mis_dict = {}
        mis_dict["Misspelled Word"] = misspelled
        fix = []
        spellcorrecter = spellcorrect(encode_dict,decode_dict,df_vocab,space_svc)
        for word in misspelled:
            out = spellcorrecter.spell_correct(word)
            fix.append(out)
        mis_dict["Did you mean"] = fix
        with st.expander("See more detail here"):
            df_mis = pd.DataFrame(mis_dict)
            st.dataframe(df_mis)
            

df_vocab ,decode_dict,encode_dict, space_svc = sc.get_resources()
segmented_vocab = df_vocab['Segmented Vocabulary'].to_list()
vocab = df_vocab['Vocabulary'].to_list()
telex_vocab = df_vocab['Telex Combination'].to_list()
add_selectbox = st.sidebar.selectbox(
    "What would you like to do ?",
    ("Introduction","Use the app", "See the source-code","How I make this project works")
    , key = 'sidebar_selection')


if st.session_state.sidebar_selection == 'Use the app':
    application()
elif st.session_state.sidebar_selection == 'Introduction':
    introduction()   




    

else:
    st.write(st.session_state.sidebar_selection)




# In[ ]:




