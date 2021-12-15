from model import SiameseBiLSTM
from inputHandler import word_embed_meta_data, create_test_data
from config import siamese_config
import pickle
import glob
from operator import itemgetter
from keras.models import load_model
from hazm import *
import os
import datetime
import glob
import itertools
import re
import json
import string
from nltk.stem.porter import *
from sklearn.feature_extraction import text
class Configuration(object):
    """Dump stuff here"""
CONFIG = Configuration()
CONFIG.embedding_dim = siamese_config['EMBEDDING_DIM']
CONFIG.max_sequence_length = siamese_config['MAX_SEQUENCE_LENGTH']
CONFIG.number_lstm_units = siamese_config['NUMBER_LSTM']
CONFIG.rate_drop_lstm = siamese_config['RATE_DROP_LSTM']
CONFIG.number_dense_units = siamese_config['NUMBER_DENSE_UNITS']
CONFIG.activation_function = siamese_config['ACTIVATION_FUNCTION']
CONFIG.rate_drop_dense = siamese_config['RATE_DROP_DENSE']
CONFIG.validation_split_ratio = siamese_config['VALIDATION_SPLIT']
subject="article"
pre_path_files =subject+"/"
stop_words_persian_path = 'Stop_Words.txt'
###### English Data ######################
def Read_Json(json_file):
    questions=[]
    with open(json_file) as f:
        data = json.load(f)
        print("i ",len(data["data"]))
        for i in range(len(data["data"])):
            print("j ",len(data["data"][i]))
            for j in range(len(data["data"][i]["paragraphs"])):
                print("k ", len(data["data"][i]))
                for k in range(len(data["data"][i]["paragraphs"][j]['qas'])):
                    question=data["data"][i]["paragraphs"][j]['qas'][k]['question']
                    questions.append(question)
                    answer=data["data"][i]["paragraphs"][j]['qas'][k]['answers']
                    print(question)
        #f = open(text_file, "a",encoding="utf-8")
        #f.write(question)
        #f.close()
def Clean_Text(st):
    st = ''.join(ch for ch in st if ch not in string.punctuation)
    st = st.strip()
    st = st.split()
    while (True):
        try:
            st.discard('')
        except:
            break
    stemmer = PorterStemmer()
    st = ' '.join([word for word in st if word not in (text.ENGLISH_STOP_WORDS)])
    st = [stemmer.stem(word) for word in st.split()]
    st = ' '.join(st)
    return st
###### Perian Data ######################
def Normalizer_Text(text):
    with open(stop_words_persian_path, 'r', encoding='utf-8') as file:
        Stop_Words = file.read()
    normalizer = Normalizer()
    compile_patterns = lambda patterns: [(re.compile(pattern), repl) for pattern, repl in patterns]
    diacritics_patterns = compile_patterns([
        ('[\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652]', ''),
        # remove FATHATAN, DAMMATAN, KASRATAN, FATHA, DAMMA, KASRA, SHADDA, SUKUN
    ])
    for pattern, repl in diacritics_patterns:
        text = pattern.sub(repl, text)
    text = normalizer.affix_spacing(text)
    text = normalizer.normalize(text)
    words = word_tokenize(text)
    temp = []
    for i in range(len(words)):
        if words[i] in Stop_Words.split():
            continue
        else:
            temp.append(words[i])
    return " ".join(temp)
def Stemmer_Text(text):
    stemmer = Stemmer()
    words = word_tokenize(text)
    temp = []
    for i in range(len(words)):
        temp.append(stemmer.stem(words[i]))
    return " ".join(temp)
##### Global ############################
def save_data(filename,data):
    with open(filename,'wb') as f:
            pickle.dump(data,f)
def load_data(filename):
    with open(filename,'rb') as f:
        return pickle.load(f)
def Create_Dataset(question,answering,languege='Persian'):
    #question = []
    #answering = []
    question_pre = []
    answering_pre = []
    is_similar=[]
    if(languege=='Persian'):
        '''
        with open(filepath, encoding='utf-8') as fp:
            line = fp.readline()
            while line:
                qa = line.strip().split('؟')[0] + '؟'
                ans = line.strip().split('؟')[1]
                qa = re.sub('\d+.', '', qa)
                question.append(qa.strip())
                if(ans not in answering):
                    answering.append(ans.strip())
                line = fp.readline()
        '''
        for i in range(len(question)):
            s_i=Normalizer_Text(question[i])
            s_i=Stemmer_Text(s_i)
            if(s_i in question_pre):
                continue
            for j in range(len(answering)):
                if(i!=j):
                    if(answering[j].strip()==answering[i].strip()):
                        continue
                s_j = Normalizer_Text(answering[j])
                s_j = Stemmer_Text(s_j)
                question_pre.append(s_i)
                answering_pre.append(s_j)
                if(i==j):
                    is_similar.append(1)
                else :
                    is_similar.append(0)
        if not os.path.exists(pre_path_files):
            os.makedirs(pre_path_files)
        if not os.path.exists(pre_path_files+'Persian/'):
            os.makedirs(pre_path_files+'Persian/')
        save_data(pre_path_files+'Persian/question_pre',question_pre)
        save_data(pre_path_files+'Persian/answering_pre', answering_pre)
        save_data(pre_path_files+'Persian/is_similar', is_similar)
        save_data(pre_path_files+'Persian/ListAnswering', answering)
        save_data(pre_path_files+'Persian/ListQuestion', question)
    elif(languege=='English'):
        '''
        with open(filepath, encoding='utf-8') as fp:
            line = fp.readline()
            while line:
                qa = line.strip().split('?')[0] + '?'
                ans = line.strip().split('?')[1]
                question.append(qa.strip())
                if (ans not in answering):
                    answering.append(ans.strip())
                line = fp.readline()
        question_pre = []
        answering_pre = []
        '''
        for i in range(len(question)):
            s_i = Clean_Text(question[i])
            for j in range(len(answering)):
                if (i != j):
                    if (answering[j].strip() == answering[i].strip()):
                        continue
                s_j = Clean_Text(answering[j])
                question_pre.append(s_i)
                answering_pre.append(s_j)
                if (i == j):
                    is_similar.append(1)
                else:
                    is_similar.append(0)
        if not os.path.exists(pre_path_files):
            os.makedirs(pre_path_files)
        if not os.path.exists(pre_path_files+'English/'):
            os.makedirs(pre_path_files+'English/')
        save_data(pre_path_files+'English/question_pre', question_pre)
        save_data(pre_path_files+'English/answering_pre', answering_pre)
        save_data(pre_path_files+'English/is_similar', is_similar)
        save_data(pre_path_files+'English/ListAnswering', answering)
        save_data(pre_path_files+'English/ListQuestion', question)
##### Train ############################
def Word_Embedding(languege='Persian'):
    file=''
    if not os.path.exists(pre_path_files+languege+'/Word2Vec'):
        os.makedirs(pre_path_files+languege+'/Word2Vec')
        file='model-'+str(datetime.datetime.now().strftime("%Y-%m-%d-%I-%M"))+'.model'
    else :
        file=glob.glob(pre_path_files+languege+'/Word2Vec/')[0]
    question_pre = load_data(pre_path_files+languege+'/question_pre')
    question_pre=[k for k, v in itertools.groupby(question_pre)]
    answering_pre = load_data(pre_path_files+languege+'/answering_pre')
    answering_pre=[k for k, v in itertools.groupby(answering_pre)]
    tokenizer, embedding_matrix = word_embed_meta_data(question_pre + answering_pre,siamese_config['EMBEDDING_DIM'],pre_path_files+languege+'/Word2Vec/'+file)
    save_data(pre_path_files+languege+'/Word2Vec/tokenizer',tokenizer)
    save_data(pre_path_files+languege+'/Word2Vec/embedding_matrix',embedding_matrix)
def Train(languege='Persian'):
        question_pre = load_data(pre_path_files+languege+'/question_pre')
        answering_pre = load_data(pre_path_files+languege+'/answering_pre')
        is_similar = load_data(pre_path_files+languege+'/is_similar')
        tokenizer = load_data(pre_path_files+languege+'/Word2Vec/tokenizer')
        embedding_matrix = load_data(pre_path_files+languege+'/Word2Vec/embedding_matrix')
        embedding_meta_data = {
            'tokenizer': tokenizer,
            'embedding_matrix': embedding_matrix
        }
        sentences_pair = [(x1, x2) for x1, x2 in zip(question_pre, answering_pre)]
        siamese = SiameseBiLSTM(CONFIG.embedding_dim, CONFIG.max_sequence_length, CONFIG.number_lstm_units,
                                CONFIG.number_dense_units, CONFIG.rate_drop_lstm, CONFIG.rate_drop_dense,
                                CONFIG.activation_function, CONFIG.validation_split_ratio)
        siamese.train_model(sentences_pair, is_similar, embedding_meta_data,
                                              model_save_directory=pre_path_files+languege+'/'+ 'Model/')
##### Update Train ############################
def Update_Word_Embedding(languege):
    if(languege==0):
        question_pre = load_data(pre_path_files+'Persian/question_pre')
        answering_pre = load_data(pre_path_files+'Persian/answering_pre')
        tokenizer, embedding_matrix = word_embed_meta_data(question_pre + answering_pre,siamese_config['EMBEDDING_DIM'])
        save_data(pre_path_files+'Persian/tokenizer',tokenizer)
        save_data(pre_path_files+'Persian/embedding_matrix',embedding_matrix)
    elif(languege==1):
        question_pre = load_data(pre_path_files+'English/question_pre')
        answering_pre = load_data('English/answering_pre')
        tokenizer, embedding_matrix = word_embed_meta_data(question_pre + answering_pre,siamese_config['EMBEDDING_DIM'])
        save_data(pre_path_files+'English/tokenizer', tokenizer)
        save_data(pre_path_files+'English/embedding_matrix', embedding_matrix)
def Update_Train(languege,question,answer):
    if (languege == 0):
        question_pre = load_data(pre_path_files + 'Persian/question_pre')
        answering_pre = load_data(pre_path_files + 'Persian/answering_pre')
        is_similar = load_data(pre_path_files + 'Persian/is_similar')
        tokenizer = load_data(pre_path_files + 'Persian/tokenizer')
        embedding_matrix = load_data(pre_path_files + 'Persian/embedding_matrix')
        embedding_meta_data = {
            'tokenizer': tokenizer,
            'embedding_matrix': embedding_matrix
        }
        sentences_pair = [(x1, x2) for x1, x2 in zip(question_pre, answering_pre)]
        siamese = SiameseBiLSTM(CONFIG.embedding_dim, CONFIG.max_sequence_length, CONFIG.number_lstm_units,
                                CONFIG.number_dense_units, CONFIG.rate_drop_lstm, CONFIG.rate_drop_dense,
                                CONFIG.activation_function, CONFIG.validation_split_ratio)
        best_model_path = siamese.train_model(sentences_pair, is_similar, embedding_meta_data,
                                              model_save_directory=pre_path_files + 'Persian/')
    elif (languege == 1):
        question_pre = load_data(pre_path_files + 'English/question_pre')
        answering_pre = load_data(pre_path_files + 'English/answering_pre')
        is_similar = load_data(pre_path_files + 'English/is_similar')
        tokenizer = load_data(pre_path_files + 'English/tokenizer')
        embedding_matrix = load_data(pre_path_files + 'English/embedding_matrix')
        embedding_meta_data = {
            'tokenizer': tokenizer,
            'embedding_matrix': embedding_matrix
        }
        sentences_pair = [(x1, x2) for x1, x2 in zip(question_pre, answering_pre)]
        siamese = SiameseBiLSTM(CONFIG.embedding_dim, CONFIG.max_sequence_length, CONFIG.number_lstm_units,
                                CONFIG.number_dense_units, CONFIG.rate_drop_lstm, CONFIG.rate_drop_dense,
                                CONFIG.activation_function, CONFIG.validation_split_ratio)
        best_model_path = siamese.train_model(sentences_pair, is_similar, embedding_meta_data,
                                              model_save_directory=pre_path_files + 'English/')
def Update_Model(languege,question,answering):
    question_pre_new = []
    answering_pre_new = []
    is_similar_new = []
    if (languege == 0):
        question_pre = load_data(pre_path_files + '/Persian/'+subject+'/question_pre')
        answering_pre = load_data(pre_path_files + '/Persian/'+subject+'/answering_pre')
        is_similar = load_data(pre_path_files + '/Persian/' + subject + '/is_similar')
        ListAnswering = load_data(pre_path_files + '/Persian/' + subject + '/ListAnswering')
        ListQuestion = load_data(pre_path_files + '/Persian/' + subject + '/ListQuestion')
        print(len(question_pre))
        for i in range(len(question)):
            if(question[i] not in ListQuestion):
                ListQuestion.append(question[i])
            s_i = Normalizer_Text(question[i])
            s_i = Stemmer_Text(s_i)
            if (s_i  in question_pre):
                continue
            for j in range(len(answering)):
                if (i != j):
                    if (answering[j].strip() == answering[i].strip()):
                        continue
                s_j = Normalizer_Text(answering[j])
                s_j = Stemmer_Text(s_j)
                question_pre_new.append(s_i)
                answering_pre_new.append(s_j)
                if (i == j):
                    if (answering[j] not in ListAnswering):
                        ListAnswering.append(answering[j])
                    is_similar_new.append(1)
                else:
                    is_similar_new.append(0)
        print(len(question_pre_new))
        print(len(question_pre))
        t_q=question_pre_new
        t_a=answering_pre_new
        for i in range(len(t_q)):
            for j in range(len(answering_pre)):
                question_pre_new.append(t_q[i])
                answering_pre_new.append(answering_pre[j])
                is_similar_new.append(0)
        for i in range(len(question_pre)):
            for j in range(len(t_a)):
                question_pre_new.append(question_pre[i])
                answering_pre_new.append(t_a[j])
                is_similar_new.append(0)
        question_pre=question_pre+question_pre_new
        answering_pre=answering_pre+answering_pre_new
        is_similar=is_similar+is_similar_new
        print(len(question_pre_new))
        print(len(question_pre))
        #save_data(pre_path_files + 'Persian/question_pre', question_pre)
        #save_data(pre_path_files + 'Persian/answering_pre', answering_pre)
        #save_data(pre_path_files + 'Persian/is_similar', is_similar)
        #save_data(pre_path_files + 'Persian/ListAnswering', answering)
        #save_data(pre_path_files + 'Persian/ListQuestion', question)
    elif (languege == 1):
        with open(filepath, encoding='utf-8') as fp:
            line = fp.readline()
            while line:
                qa = line.strip().split('?')[0] + '?'
                ans = line.strip().split('?')[1]
                question.append(qa.strip())
                if (ans not in answering):
                    answering.append(ans.strip())
                line = fp.readline()
        question_pre = []
        answering_pre = []
        for i in range(len(question)):
            s_i = Clean_Text(question[i])
            for j in range(len(answering)):
                if (i != j):
                    if (answering[j].strip() == answering[i].strip()):
                        continue
                s_j = Clean_Text(answering[j])
                question_pre.append(s_i)
                answering_pre.append(s_j)
                if (i == j):
                    is_similar.append(1)
                else:
                    is_similar.append(0)
        save_data(pre_path_files + 'English/question_pre', question_pre)
        save_data(pre_path_files + 'English/answering_pre', answering_pre)
        save_data(pre_path_files + 'English/is_similar', is_similar)
        save_data(pre_path_files + 'English/ListAnswering', answering)
        save_data(pre_path_files + 'English/ListQuestion', question)
##### Question ############################
def Ask_Question(question,languege='Persian'):
    if(languege=='Persian'):
        question = Stemmer_Text(Normalizer_Text(question))
        Answers = load_data(pre_path_files+'Persian/ListAnswering')
        tokenizer = load_data(pre_path_files+'Persian/Word2Vec/tokenizer')
        embedding_matrix = load_data(pre_path_files+'Persian/Word2Vec/embedding_matrix')
        embedding_meta_data = {
            'tokenizer': tokenizer,
            'embedding_matrix': embedding_matrix
        }
        model_path_persian=glob.glob(pre_path_files+'Persian/Model/*.h5')[0]
        model = load_model(model_path_persian)
        test_sentence_pairs = []
        for i in range(len(Answers)):
            test_sentence_pairs.append((question, Stemmer_Text(Normalizer_Text(Answers[i]))))
        test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer, test_sentence_pairs,
                                                                  siamese_config['MAX_SEQUENCE_LENGTH'])
        preds = list(model.predict([test_data_x1, test_data_x2, leaks_test], verbose=1).ravel())
        results = [z for (x, y), z in zip(test_sentence_pairs, preds)]
        out = sorted(range(len(results)), key=lambda k: results[k])
        result_list = []
        for j in range(5):
            result_list.append({'answer': Answers[out[len(out) - 1 - j]],
                                'score': str(results[out[len(out) - 1 - j]])})
        return result_list

    elif(languege=='English'):
        question = Clean_Text(question)
        Answers = load_data(pre_path_files+'English/ListAnswering')
        print(Answers)
        tokenizer = load_data(pre_path_files+'English/Word2Vec/tokenizer')
        embedding_matrix = load_data(pre_path_files+'English/Word2Vec/embedding_matrix')
        embedding_meta_data = {
            'tokenizer': tokenizer,
            'embedding_matrix': embedding_matrix
        }
        model_path_english=glob.glob(pre_path_files+'English/Word2Vec/*.h5')[0]
        model = load_model(model_path_english)
        test_sentence_pairs = []
        for i in range(len(Answers)):
            test_sentence_pairs.append((question, Clean_Text(Answers[i])))
        test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer, test_sentence_pairs,
                                                                  siamese_config['MAX_SEQUENCE_LENGTH'])
        preds = list(model.predict([test_data_x1, test_data_x2, leaks_test], verbose=1).ravel())
        results = [z for (x, y), z in zip(test_sentence_pairs, preds)]
        out = sorted(range(len(results)), key=lambda k: results[k])
        result_list = []
        for j in range(5):
            result_list.append({'answer': Answers[out[len(out) - 1 - j]],
                                'score': str(results[out[len(out) - 1 - j]])})
        return result_list
def Accuracy():
    Questions = load_data('ListQuestion')
    Answering = load_data('ListAnswering')
    co=0
    for i in range(len(Questions)):
        ans=Ask_Question(Questions[i])
        print(i)
        if(ans[0]==Answering[i]):
            co=co+1
        else:
            print(Questions[i])
            print(ans[0])
            print(Answering[i])
        print("------------------------")
    print("Accuracy on Train ",co/len(Questions))
