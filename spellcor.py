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

  def decode_output(self,text):
    telex_init = 'aieouyd'
    telex_next = 'sfwxjraeod'
    telex_accent = 'sfwxj'
    text = text.lower()
    word_list = text.split()
    decoded_txt = []
    for word in word_list:
      decoded_word = []
      decoded = ''
      for i in range(len(word)):
        pad = ' '
        word += pad
        char = word[i]
        if char not in decoded:
          if char in telex_init:
            i += 1
            next_char = word[i]
            index = i
            if next_char in telex_next:
              char = char + next_char
              if char not in self.decode_dict.keys():
                char = word[i-1]
              else:
                i +=1
                if word[i] in telex_accent and char + word[i] in self.decode_dict.keys():
                    char += word[i]

            decoded = char
          decoded_word.append(self.decode_dict[char])

      decoded_txt.append(''.join(decoded_word))
    return ' '.join(decoded_txt)

  def spell_correct(self,text,output = 'word',force_correct = False):
    if output not in ['word','similarity','DataFrame','hybrid']:
      print("Error")
    else:
      text = text.lower()
      is_space = (self.space_svc.predict(self.encode_word_for_svc(text)) == 1)
      if not is_space or force_correct:
            en_txt = self.encode_input(text)
            sim = [td.DamerauLevenshtein(qval=1).normalized_similarity(en_txt,str(word)) for word in self.telex_vocab]
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

  def first_split(self,text):
    telex_comb_list = []
    index_in_str = []
    for key in self.decode_dict.keys():
      if key in text and len(key) > 1:
        telex_comb_list.append(key)
        index_in_str.append(text.index(key))
    first_comb = min(index_in_str)
    
    if index_in_str.count(first_comb) != 1:
      index = np.where(np.array(index_in_str) == first_comb)[0]
      comb_list = [telex_comb_list[i] for i in index]
      split_letter = max(comb_list,key=len)
    else:
      index = np.where(np.array(index_in_str) == first_comb)[0][0]
      split_letter = telex_comb_list[index]

    split_text = text.split(split_letter)
    split_text[0] = split_text[0] + split_letter 
    return " ".join(split_text)

  def latter_split(self,text):
    space = text.index(' ')
    if space + 1 == (len(text) - 2):
      return None, True
    else:
      after_space = text[space + 1]
      out = list(text)
      out[space] = after_space
      out[space + 1] = ' '
      return ''.join(out), False

  def split_word(self,text,threshold = 0.85):
    text = text.lower()
    split_text = self.first_split(text)
    sim_pair = []
    sim_word = []
    st.write(split_text)
    for word in split_text.split():

      sim = self.spell_correct(word,output='hybrid',force_correct=True) 

      sim_pair.append(sim[0][1])
      sim_word.append(sim[0][0])
    sim_rec = []
    sim_rec.append([sim_pair,sim_word])
    mean_sim = []
    mean_sim.append(np.mean(sim_pair))
    if np.mean(sim_pair) >= threshold:
      best_pair = sim_rec[0][1]
      return " ".join(best_pair)
    else:
      flag = False
      while np.mean(sim_pair) <= threshold and flag == False:
        next_split, flag = self.latter_split(split_text)  
        sim_pair = []
        sim_word = []
        for word in next_split.split():
          sim = self.spell_correct(word,output='both') 
          sim_pair.append(sim[0][1])
          sim_word.append(sim[0][0])
        sim_rec.append([sim_pair,sim_word])
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
    alphabet = [' ', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'á', 'à', 'ả', 'ã', 'ạ', 'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ', 'ó', 'ò', 'ỏ', 'õ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ', 'é', 'è', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ', 'ú', 'ù', 'ủ', 'ũ', 'ụ', 'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự', 'í', 'ì', 'ỉ', 'ĩ', 'ị', 'ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ', 'đ', 'ươ', 'ướ', 'ưở', 'ườ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    telex = [' ', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'as', 'af', 'ar', 'ax', 'aj', 'aa', 'aas', 'aaf', 'aar', 'aax', 'aaj', 'aw', 'aws', 'awf', 'awr', 'awx', 'awj', 'os', 'of', 'or', 'ox', 'oj', 'oo', 'oos', 'oof', 'oor', 'oox', 'ooj', 'ow', 'ows', 'owf', 'owr', 'owx', 'owj', 'es', 'ef', 'er', 'ex', 'ej', 'ee', 'ees', 'eef', 'eer', 'eex', 'eej', 'us', 'uf', 'ur', 'ux', 'uj', 'uw', 'uws', 'uwf', 'uwr', 'uwx', 'uwj', 'is', 'if', 'ir', 'ix', 'ij', 'ys', 'yf', 'yr', 'yx', 'yj', 'dd', 'uow', 'uows', 'uowr', 'uowf', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    telex_to_letter = {}
    for i in range(len(alphabet)):
        telex_to_letter[telex[i]]=alphabet[i]
    letter_to_telex = {}
    for i in range(len(alphabet)):
        letter_to_telex[alphabet[i]]=telex[i]
    return df_vocab, telex_to_letter, letter_to_telex, space_svc