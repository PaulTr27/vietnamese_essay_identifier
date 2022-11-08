# Vietnamese Essay Categorizer
An educational tool for young students that helps with writing essays
## Description
### Overview
- At primary levels of education in Vietnam, students are introduced to 5 categories of essays:
  - Argumentative - Nghị luận
  - Expressive - Biểu cảm
  - Descriptive - Miêu tả
  - Narrative - Tự sự
  - Expository - Thuyết minh

- New students usually find it hard to write the required category (E.g. Students describe too much in an expressive essay).
- This AI assistant includes a Fine-tuned SequenceClassificationPhoBERT model for classifying categories of input essay and a self written SpellCorrect algorithm for spell-checking and alternative-suggesting.
- For more information about this project, you can go to my [slides](https://hackmd.io/@Froggyplayz123/DL4AI_Presentation#/3) or you can see more in [my streamlit app](https://paultran2734-vietnamese-essay-i-vietnameseessayassistant-kya1wv.streamlitapp.com/). *VNcoreNLP has updated a new python wrapper and the app in this repo are currently outdated*
- You can also find the model used in this App on [Huggingface](https://huggingface.co/PaulTran/vietnamese_essay_identify)

# Dependencies

In your terminal or command prompt, clone this repository:
```
git clone https://github.com/PaulTran2734/vietnamese_essay_identifier/
```
change your working directory:
```
cd vietnamese_essay_identifier
```
then install requirements using pip:
```
pip install -r requirements.txt
```
To run streamlit locally, use:
```
streamlit run VietnameseEssayAssistant.py
```
