from vncorenlp import VnCoreNLP
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
from annotated_text import annotated_text, annotation
from io import BytesIO
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
import matplotlib.pyplot as plt

st.set_page_config(page_title = "Vietnam EduNLP", 
                   page_icon = 'https://i.imgur.com/BnJxvi3.png',
                   layout = 'wide'
                   )

@st.cache(allow_output_mutation=True,show_spinner=False)
def load_model(model_path,seg_path):
    segmenter = VnCoreNLP(seg_path, annotators="wseg", max_heap_size='-Xmx500m') 
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer, segmenter
    # In[6]:
def introduction():
    markdown = "# ***WELCOME TO YOUR ESSAY ASSISTANT*** \n ## How to use this web-app \n Hi! This is Paul, the author of this project. To use the app, please select 'Use the application' on the sidebar to your left. If you want more information about this project or the source-code, please check out my [GitHub repository](https://github.com/PaulTran2734/vietnamese_essay_identifier) \n ## About this application \n - This is a handy NLP web-app created for those who struggles to write the right type of essay in Vietnamese. Especially students of primary or secondary level of education in Vietnam. \n - At primary level of education in Vietnam, students are introduced to 5 types of essay: Argumentative, Espressive, Expository, Narrative, Descriptive. When they are first introduced to these, they can find it hard to write the right type of essay. For example, they can be too *descriptive* in an **Expressive** essay which requires more emotion or thoughts about a subject or an object than the specifity of its features. Therefore, this simple application was developed in order to solve this problem to a certain degree. \n- A spell-check algorithm is also included in this app. Despite its inefficiency, it can at least check for words that are mistyped, or not exist in Vietnamese Dictionary. This is my first ever attempt in writing an NLP algorithm, hence its inefficiency. \n ## More about my project \n To know more about my project(e.g. which model I used and how I trained my model,etc) , go to 'How I make this project works' tab on the sidebar to your left. :smile:"
    st.markdown(markdown)
def radio_callback():
    st.session_state.radio_option = st.session_state.radio
def resubmit():
    st.session_state.fixed = True
    st.session_state.phase = 2
def reset():
    st.session_state.fixed = False
    st.session_state.phase = 1

def application(): 
    st.header("Welcome to your friendly app")
    cwd = os.getcwd() 
    segmenter_path = os.path.join(cwd,'VnCoreNLP-1.1.1.jar')
    with st.spinner(text="Initializing..."):
        classifier, tokenizer, segmenter = load_model("PaulTran/vietnamese_essay_identify",                                        
                                                      segmenter_path)
    if "fixed" not in st.session_state:
        st.session_state.fixed = False
    if 'phase' not in st.session_state:
        st.session_state.phase = 1

    if st.session_state.phase == 1:
        notes = st.empty() 
        notes.markdown('#### Write your essay in the text box and click Submit to see the results')
        text_area = st.empty()
        essay = text_area.text_area('Type your essay here', 'Hãy viết bài văn của bạn vào đây',key = 'input_text',height=600)
        
        placeholder = st.empty()
        isclick = placeholder.button('Submit')
        if isclick:

            st.session_state.phase += 1
            placeholder.empty()
            text_area.empty()
            notes.empty()


    if st.session_state.phase == 2:
        fixed = False
        sentences_list = st.session_state.input_text.split('.')
        pred_labels_list = []
        misspelled = []
        label_names = ['Nghị Luận','Biểu cảm','Miêu tả','Tự sự', 'Thuyết minh']
        with st.spinner("Processing"):
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
                    softmax_nn = torch.nn.Softmax(dim=1)
                    softmax = softmax_nn(logits)    
                    prediction = torch.argmax(softmax, dim=1)
                    pred_labels_list.append(prediction)
            pred_prob = []
            for i in range(5):
                pred_prob.append((pred_labels_list.count(i)/len(pred_labels_list)))
            pred_label_name = label_names[np.argmax(pred_prob)]

        if len(misspelled) == 0:
            st.session_state.phase = 3
        elif len(misspelled) > 0 and not st.session_state.fixed:
            with st.spinner("Evaluating..."):
                fix = []
                spellcorrecter = spellcorrect(encode_dict,decode_dict,df_vocab,space_svc)
                for word in misspelled:
                    out = spellcorrecter.spell_correct(word)
                    fix.append(out)
                to_ann = st.session_state.input_text
                ann_list = [txt + ' ' for txt in list(to_ann.split(' '))]
                for word in misspelled:
                    if (word +' ') in ann_list:
                        idx = ann_list.index(word +' ')
                        suggest = fix[misspelled.index(word)]
                        ann_list[idx] = (word,suggest,'#faa') 
                if not any([isinstance(value,list) for value in ann_list]):
                    st.session_state.phase = 3 
                else:
                    col1, col2 = st.columns(2)
                    col1.text_area("Fix your essay here",st.session_state.input_text,key = 'input_text',height = 1000)
                    col2.write("Detected mistakes with suggestions")
                    with col2:
                        st.markdown(f',<p align ="justify">{annotated_text(*ann_list)}</p>',allow_unsafe_html = True)
                    st.write("Click resubmit to submit again after you fixed the mistakes")
                    resubmit_button = st.button('Resubmit',on_click=resubmit())

        if st.session_state.phase == 3 or st.session_state.fixed:
            col1,col2 = st.columns([10,7])
            with col1:
                st.markdown("## Your essay")
                st.write(st.session_state.input_text)
            with col2:
                st.markdown("## Evaluation")

                md_txt = f"You have writen a/an **{pred_label_name}** essay."
                st.markdown(md_txt)
            
            with col2:
                    fig, ax = plt.subplots(figsize=(7, 3))
                    fig.suptitle('How your essay sounds', fontsize=16)
                    wedges, texts, autotexts = ax.pie(pred_prob, autopct='%1.1f%%',
                            shadow=False, startangle=90,radius = 2 )

                    ax.axis('equal')  
                    ax.legend(wedges,label_names,title = "Categories",title_fontsize = 'x-large',fontsize = "medium")
                    plt.setp(autotexts, size=10, weight="bold")
                    buf = BytesIO()
                    fig.savefig(buf, format="png")
                    st.image(buf)
                    st.write(f'Word count: {len(st.session_state.input_text.split())}')
            back = st.empty()
            go_back = back.button("Return",on_click = reset())
            if go_back:
                back.empty()
        

def project_detail():
    
        cwd = os.getcwd()
        segmenter_path = os.path.join(cwd,'VnCoreNLP','VnCoreNLP-1.1.1.jar')
        segmenter = VnCoreNLP(segmenter_path, annotators="wseg", max_heap_size='-Xmx500m') 
        st.header("Click on expander for more information")
        st.markdown("#### ***Important***  \n  *Read 'Global variables' before continue!!*")
        with st.expander("Global variables"):
            st.markdown("-  ***encode_dict*** and ***decode_dict***: Dictionaries with letter as keys and telex combination as values and vice versa")
            st.markdown("-  ***df_vocab***: A dataframe with Vocabulary, segmented vocabulary and telex combination of segmented vocabulary for calculating distance")
            st.dataframe(df_vocab[20000:20005])
            st.markdown("-  ***classifier***, ***tokenizer***: Finedtuned model and its tokenizer   \n  -   ***segmenter***: VNCoreNLP word segmenter  \n  -    ***sc***: Imported self-written spellcor.py module  \n  -   ***spellcorrector***: spellcorrect Class in spellcor.py.   \n  That's about it, you can close this expander and proceed :smile:   ")
        with st.expander('How my spell-check algorithm works:'):
            st.markdown('      The source-code is located on [GitHub](https://github.com/PaulTran2734/vietnamese_essay_identifier/blob/main/spellcor.py)')
            st.markdown('### About the algorithm  \n The general baseline is quite simple. The algorithm first finds the mistakes in the essay, then calculates the similarity of mistaken words with every single word in a 74k words dictionary using DamerauLevenstein algorithm of [textdistance](https://github.com/life4/textdistance) module, and finally suggests alternatives. Quite simple right! :wink: ')
            st.markdown('Now, we move on to the codes. Let our example string of text be: "ddaay flaf một câu sai chísnh tar và cần sữa chữa" ')
            with st.echo():
                sentence = "ddaay flaf một câu sai chísnh tar và cần sữa chữa" # misspell-filled sentence
                correct_sentence = "Đây là một câu sai chính tả và cần sửa chữa" # correct sentence
            st.markdown('You can see the mistakes in the example. The first action of the algorithm is to find mistakes in the segmented sentences (see in the model expander).')
            with st.echo():
                misspelled = [] # a list contains mistakes
                segment = segmenter.tokenize(sentence)
                # print(segment)
                # segment = [['ddaay', 'laf', 'một', 'câu', 'sai', 'chisnh', 'tar']]
                if len(segment) > 0: # only execute if the sentence is not empty
                    for word in segment[0]:
                        if word not in segmented_vocab and word not in decode_dict.keys():
                            misspelled.append(word)      
            st.markdown("After finding and adding all the mistakes in the sentence, the mistakes are pass through an algorithm:")
            with st.echo():
                # encoded_mistake = [] # demo-purposed list
                suggest = []
                sim = []
                for mistake in misspelled: 
                    text = mistake.lower() #lower-case for encoding
                    encoded_text = spellcorrecter.encode_input(text)
                    #    encoded_mistake.append(encoded_text)
                #   print(encoded_mistake) 
                #   ["ddaay","laf","chissnh","tar"]

                    d_lev = td.DamerauLevenshtein(qval=1) # Initialize algorithm from textdistance
                    similarity = [d_lev.normalized_similarity(encoded_text,str(word)) 
                                  for word in telex_vocab] # calculate similarity
                    df_results = pd.DataFrame(vocab) # Dataframe for easier data accessing
                    df_results['Similarity'] = similarity
                    corrected = df_results.sort_values(['Similarity'], ascending=False)
                    suggest.append(corrected.values[0][0])
                    sim.append(corrected.values[0][1]) # Word with highest similarity
            st.markdown("We have successfully found alternatives, now we can view the results using pandas Dataframe:")
            df_suggest = pd.DataFrame()
            df_suggest['Mistakes'] = misspelled
            df_suggest["Suggestions"] = suggest
            df_suggest["Similarity"]  = sim
            st.dataframe(df_suggest)       
            st.markdown("We can see that my algorithm works great with simple typo ,but it can not detects and provides alternatives for word in context (sửa and sữa) as the wrong word exists in the dictionary.")
            st.markdown("That is it. This is my first ever self-written NLP algorithm. I will continue on my path to develope a better Spell Correcting algorithm")

        with st.expander("How finetuned my model:"):
            st.markdown("My model is a finetuned [PhoBERT model](https://huggingface.co/vinai/phobert-base) for a down-stream task of classify sentences to 5 categories.")
            st.markdown("I will only go through how I acquired and processed the data. \n You can check out how to use my model on [Huggingface](https://huggingface.co/PaulTran/vietnamese_essay_identify)")
            st.markdown("#### Data acquiring \n  The dataset was made from various sources of sample essays on the internet. \n The essays is split to sentences with respective labels as you can see here in the dataframe: ")
            df_dataset = pd.read_csv(os.path.join(cwd,"dataset.csv"))
            st.dataframe(df_dataset)
            st.markdown("____")
            st.markdown("-  With the help of train_test_split function of sklearn module, I splitted the dataset to 80% train set, 10% validation set and 10% test set.")
            st.markdown("-  PhoBERT will not work well if words were not segmented. Therefore, I use [VNCoreNLP word segmenter](https://github.com/vncorenlp/VnCoreNLP) to segment inputs before training the model.")
            st.write("- An example from VNCoreNLP GitHub repository:")
            st.code(''' # Input 
text = "Ông Nguyễn Khắc Chúc  đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."

# To perform word segmentation, POS tagging, NER and then dependency parsing
annotated_text = annotator.annotate(text)   
# To perform word segmentation only
word_segmented_text = annotator.tokenize(text)

print(word_segmented_text)


>>> [['Ông', 'Nguyễn_Khắc_Chúc', 'đang', 'làm_việc', 'tại', 'Đại_học', 'Quốc_gia', 'Hà_Nội', '.'], ['Bà', 'Lan', ',', 'vợ', 'ông', 'Chúc', ',', 'cũng', 'làm_việc', 'tại', 'đây', '.']]

'''
)

            st.markdown("Lastly, the segmented texts were tokenized with Transformers-model's Tokenizer, in this case PhoBERT model's Tokenizer.")
            st.markdown("After I has done processed the training dataset, I finetuned the model in 10 epochs, thus created the model for this project. \n That is all about my model. ")


            
if __name__ == '__main__':
    if 'sidebar_selection' not in st.session_state:
        st.session_state.sidebar_selection = 'Introduction'
    df_vocab ,decode_dict,encode_dict, space_svc = sc.get_resources()
    segmented_vocab = df_vocab['Segmented Vocabulary'].to_list()
    vocab = df_vocab['Vocabulary'].to_list()
    telex_vocab = df_vocab['Telex Combination'].to_list()
    spellcorrecter = sc.spellcorrect(encode_dict,decode_dict,df_vocab,space_svc)
    add_selectbox = st.sidebar.selectbox(
        "What would you like to do ?",
        ("Introduction","Use the app", "How I make this project works")
        , key = 'sidebar_selection')


    if st.session_state.sidebar_selection == 'Use the app':
        application()
    elif st.session_state.sidebar_selection == 'Introduction':
        introduction()   
    else:
        project_detail()


# In[ ]:


