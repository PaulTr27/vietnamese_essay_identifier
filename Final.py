from vncorenlp import VnCoreNLP
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
from annotated_text import annotated_text, annotation
import os
 


import streamlit as st
import numpy as np
@st.cache(allow_output_mutation=True,show_spinner=False)
def load_model(model_path,tokenizer_path,seg_path):
    segmenter = VnCoreNLP(seg_path, annotators="wseg", max_heap_size='-Xmx500m') 
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer, segmenter
    # In[6]:

 

st.title('Testing function app')



add_selectbox = st.sidebar.selectbox(
    "What would you like to do ?",
    ("Use the app", "See the source-code")
)
if add_selectbox == 'Use the app':
    st.write("Welcome to your friendly app")
    cwd = os.getcwd() 
    seg_path = os.path.join(cwd,'VnCoreNLP/VnCoreNLP-1.1.1.jar')
    with st.spinner(text="Initializing..."):
        classifier, tokenizer, segmenter = load_model("PaulTran/vietnamese_essay_identify",
                                                      "vinai/phobert-base",
                                                      seg_path)
    essay = st.text_area('Write your essay here', "Đây là ví dụ",height=600)
    sentences_list = essay.split('.')

    submission = st.button('Submit')
    if submission:
        pred_labels_list = []
        label_names = ['Nghị Luận','Biểu cảm','Miêu tả','Tự sự', 'Thuyết minh']

        for sentence in sentences_list:
            sentence = sentence.lower()
            segment = segmenter.tokenize(sentence)
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
    with st.expander("See more detail here"):
        txt = annotated_text((essay,"#8ef"))
        st.write(txt)




        

else:
    st.error("Nothing just yet")




# In[ ]:




