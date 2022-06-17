from base64 import encode
from cmath import nan
import textdistance as td
import pandas as pd
import os
import joblib 
import streamlit as st
import numpy as np
class spellcorrect:
  def __init__(self,encode_dict,decode_dict,df_vocab,space_svc):
    self.space_svc = space_svc
    self.encode_dict = encode_dict
    self.decode_dict = decode_dict
    self.df_vocab = df_vocab
    self.segmented_vocab = df_vocab['Segmented Vocabulary'].values.tolist()
    self.vocab = df_vocab['Vocabulary'].values.tolist()
    self.telex_vocab = df_vocab['Telex Combination'].values.tolist()
  def encode_input(self,text):
    text = text.lower()
    encoded_txt = []
    for char in text:
      telex_char = self.encode_dict[char]
      encoded_txt.append(str(telex_char))
    return ''.join(encoded_txt)

  def spell_correct(self,text,output = 'word',force_correct = False,qval=2):
    if output not in ['word','similarity','DataFrame','hybrid']:
      print("Error")
    else:
      text = text.lower()
      is_space = (self.space_svc.predict(self.encode_word_for_svc(text)) == 1)
      if not is_space or force_correct:
            en_txt = self.encode_input(text)
            sim = [td.DamerauLevenshtein(qval=qval).normalized_similarity(en_txt,str(word)) for word in self.telex_vocab]
            df_res = pd.DataFrame(self.vocab)
            df_res['Similarity'] = sim
            corrected = df_res.sort_values(['Similarity'], ascending=False)
            if output == 'word':
              return corrected.values[0][0]
            if output == 'similarity':
              return corrected.values[0][1]
            if output == 'hybrid':
              return corrected.values
            else:
              return corrected.head()
      
      else:
        return self.split_word(text)

  

  def split_txt(self,text):
    if ' ' in text:
      space = text.index(' ')
    else:
      space = 0
    if space + 1 == (len(text) - 2):
      return None, True
    else:
      if text[space] == text[-1]:
        return text[:-2], True
      after_space = text[space + 1]
      out = list(text)
      out[space] = after_space
      out[space + 1] = ' '
      return ''.join(out), False

  def split_word(self,text,threshold = 0.85):
    text = text.lower()
    split_text, flag = self.split_txt(' '+text)
    sim_pair = []
    word_pair = []

    for word in split_text:

        sim = self.spell_correct(word,output='hybrid',force_correct=True,qval=1) 
        sim_pair.append(sim[0][1])
        word_pair.append(sim[0][0])

    sim_rec = []
    sim_rec.append([sim_pair,word_pair])
    mean_sim = []
    mean_sim.append(np.mean(sim_pair))
    if np.mean(sim_pair) >= threshold:
      best_pair = sim_rec[0][1]
      return " ".join(best_pair)
    else:
      next_split = split_text
      flag = False
      while np.mean(sim_pair) <= threshold and flag == False:
        next_split, flag = self.split_txt(next_split)  
        sim_pair = []
        word_pair = []
        st.write(next_split)
        for word in next_split.split():
          sim = self.spell_correct(word,output='hybrid') 
          sim_pair.append(sim[0][1])
          word_pair.append(sim[0][0])
        sim_rec.append([sim_pair,word_pair])
        mean_sim.append(np.mean(sim_pair))
      best_pair = sim_rec[np.argmax(mean_sim)][1]
      return " ".join(best_pair)
  def encode_word_for_svc(self,text):
    MAXLEN = 24
    encoded_word = []
    for char in text:
      encoded_word.append(list(self.encode_dict.keys()).index(char))

    if len(encoded_word) < MAXLEN:
        encoded_word = np.concatenate([encoded_word,np.zeros(MAXLEN - len(encoded_word))])
    return np.reshape(encoded_word,(1,-1))


def get_resources():
    cwd = os.getcwd()
    df_vocab_path = os.path.join(cwd,'TextCorrect','Vocabulary.csv')
    space_svc= joblib.load(os.path.join(cwd,'TextCorrect','svc_space.joblib'))
    df_vocab = pd.read_csv(df_vocab_path)
    alphabet = [' ', '_','-','–', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'á', 'à', 'ả', 'ã', 'ạ', 'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ', 'ó', 'ò', 'ỏ', 'õ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ', 'é', 'è', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ', 'ú', 'ù', 'ủ', 'ũ', 'ụ', 'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự', 'í', 'ì', 'ỉ', 'ĩ', 'ị', 'ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ', 'đ', 'ươ', 'ướ', 'ưở', 'ườ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    telex = [' ', '_','-','–', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'as', 'af', 'ar', 'ax', 'aj', 'aa', 'aas', 'aaf', 'aar', 'aax', 'aaj', 'aw', 'aws', 'awf', 'awr', 'awx', 'awj', 'os', 'of', 'or', 'ox', 'oj', 'oo', 'oos', 'oof', 'oor', 'oox', 'ooj', 'ow', 'ows', 'owf', 'owr', 'owx', 'owj', 'es', 'ef', 'er', 'ex', 'ej', 'ee', 'ees', 'eef', 'eer', 'eex', 'eej', 'us', 'uf', 'ur', 'ux', 'uj', 'uw', 'uws', 'uwf', 'uwr', 'uwx', 'uwj', 'is', 'if', 'ir', 'ix', 'ij', 'ys', 'yf', 'yr', 'yx', 'yj', 'dd', 'uow', 'uows', 'uowr', 'uowf', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    telex_to_letter = {}
    for i in range(len(alphabet)):
        telex_to_letter[telex[i]]=alphabet[i]
    letter_to_telex = {}
    for i in range(len(alphabet)):
        letter_to_telex[alphabet[i]]=telex[i]
    return df_vocab, telex_to_letter, letter_to_telex, space_svc
