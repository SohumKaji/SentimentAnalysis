import os
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


train_path = "../resource/asnlib/public/aclImdb/train/" # use terminal to ls files under this directory
test_path = "../resource/asnlib/public/imdb_te.csv" # test data for grade evaluation


def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
 #output_file = csv.writer(open(outpath+name, 'wb'))
 count = 0
 agg_temp_buffer = []
 agg_scores = []

 ###################
 #List of StopWords#
 ###################
 stop_words_file = open('stopwords.en.txt', 'r')

 stop_words = stop_words_file.readline()
 stop_words_list = []
 while(stop_words):
  stop_words_list.append(stop_words[:-1])
  stop_words = stop_words_file.readline()


 #print stop_words_list

 stop_words_file.close() 
  
 ############### 
 #Files in /pos#
 ############### 
 
 for filename in os.listdir(inpath+'pos/'):
  temp_file = open(inpath+'pos/'+filename, 'r')
  temp_buffer = temp_file.read()
  
  
  
  agg_temp_buffer.append(temp_buffer)
  agg_scores.append(1)
  #print count, "|", filename
  temp_file.close()
  count +=1


 ############### 
 #Files in /neg#
 ############### 
  
 for filename in os.listdir(inpath+'neg/'):
  temp_file = open(inpath+'neg/'+filename, 'r')
  temp_buffer = temp_file.read()
  agg_temp_buffer.append(temp_buffer)
  agg_scores.append(0)
  #print count, "|", filename
  temp_file.close()
  count +=1
 
 ##################
 #Remove Stopwords#
 ##################
 
 final_buffer = []
 #only_once = 0
 for x in agg_temp_buffer:
   
  temp_ = re.split('''[!?.,'"<> ]''', x)
  temp_ = [word for word in temp_ if word !='']
  result = [words for words in temp_ if words.lower() not in stop_words_list]
  final_result = ' '.join(result)
  final_buffer.append(final_result)
  #if(only_once == 100): print x; print temp_; print result; print final_result
  #only_once +=1 
 
 
 
 ####################
 #Create imdb_tr.csv#
 ####################
 
 csv_output = {'text':final_buffer,
                'polarity':agg_scores}
 df = pd.DataFrame(csv_output, columns = ['text', 'polarity']) 
 df.to_csv(outpath+name)
 
 ######## 
 #Output#
 ########
 """
 ef = pd.read_csv(outpath+name)
 print ef.shape
 print ef
 
 tf = pd.read_csv('imdb_tr.sample.csv')
 print tf
 """
 pass

if __name__ == "__main__":

###############
#Preprocessing#
###############
 
 #imdb_data_preprocess(train_path)
 
####################
#Unigram.output.txt#
####################
 
 df = pd.read_csv('imdb_tr.csv')
 vectorizer = CountVectorizer(min_df=1)
 corpus = []
 answers= []

 for document in df.text:
  corpus.append(document)

 for polar in df.polarity:
  answers.append(polar)
 
 
 X = vectorizer.fit_transform(corpus)
 y = answers

 #print len(vectorizer.get_feature_names())
 
 #SGD = SGDClassifier(loss="hinge", penalty="l2")
 #parameters = {'learning_rate': ['optimal', 'constant', 'invscaling'], 'eta0':[.226, .2265, .227]}
 #clf = GridSearchCV(SGD, parameters, cv = 5)
 
 clf = SGDClassifier(loss="hinge", penalty="l2", learning_rate = 'invscaling', eta0 = .2265)
 clf.fit(X, y)

 test = vectorizer.transform([df.text[1]])

 #print clf.predict(test)
 #print df.polarity[1]
 
 
 #print clf.score(X,y)
 #print clf.best_score_
 #print clf.best_params_
 
 #Testing
 test_df = pd.read_csv(test_path, encoding='cp850')
 
 
 unigram_answer = []
 out_file1 = open('unigram.output.txt', 'w')
 
 #only_once = 0
 for document in test_df.text:
  X_test = vectorizer.transform([document])
  out_file1.write(str(clf.predict(X_test)[0])+'\n')

 
 out_file1.close()
 
 print "Unigram - Done!"
 
###################
#Bigram.output.txt#
###################
 
 #df = pd.read_csv('imdb_tr.csv')
 vectorizer2 = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
 #corpus = []
 #answers= []

 #for document in df.text:
 # corpus.append(document)

 #for polar in df.polarity:
 # answers.append(polar)
 
 
 X2 = vectorizer2.fit_transform(corpus)
 y2 = answers

 #print len(vectorizer.get_feature_names())


 
 #SGD = SGDClassifier(loss="hinge", penalty="l2")
 #parameters = {'learning_rate': ['optimal', 'constant', 'invscaling'], 'eta0':[.225, .226, .227]}
 #clf = GridSearchCV(SGD, parameters, cv = 5)
 
 clf2 = SGDClassifier(loss="hinge", penalty="l2", learning_rate = 'invscaling', eta0 = .225)
 clf2.fit(X2, y2)

 #test = vectorizer.transform([df.text[1]])

 #print clf.predict(test)
 #print df.polarity[1]
 
 
 #print clf.score(X,y)
 #print clf.best_score_
 #print clf.best_params_
 
 #Testing
 test_df2 = pd.read_csv(test_path, encoding='cp850')
 
 
 bigram_answer2 = []
 out_file2 = open('bigram.output.txt', 'w')
 
 #only_once = 0
 for document in test_df2.text:
  X_test2 = vectorizer2.transform([document])
  out_file2.write(str(clf2.predict(X_test2)[0])+'\n')

 
 out_file2.close()
 
 print "Bigram - Done!"

#########################
#unigramtfidf.output.txt#
#########################
 
 #df = pd.read_csv('imdb_tr.csv')
 vectorizer3 = TfidfVectorizer(min_df=1)
 #corpus = []
 #answers= []

 #for document in df.text:
 # corpus.append(document)

 #for polar in df.polarity:
 # answers.append(polar)
 
 
 X3 = vectorizer3.fit_transform(corpus)
 y3 = answers

 #print len(vectorizer.get_feature_names())


 
 SGD3 = SGDClassifier(loss="hinge", penalty="l2")
 parameters = {'learning_rate': ['optimal', 'constant', 'invscaling'], 'eta0':[.3,.30625 ,.3125]}
 clf3 = GridSearchCV(SGD3, parameters, cv = 5)
 
 #clf3 = SGDClassifier(loss="hinge", penalty="l2", learning_rate = 'invscaling', eta0 = .30625)
 clf3.fit(X3, y3)

 test = vectorizer3.transform([df.text[1]])

 #print clf3.predict(test)
 #print df.polarity[1]
 
 
 #print clf3.score(X3,y3)
 #print clf3.best_score_
 #print clf3.best_params_
 
 #Testing
 test_df3 = pd.read_csv(test_path, encoding='cp850')
 
 
 bigram_answer3 = []
 out_file3 = open('unigramtfidf.output.txt', 'w')
 
 #only_once = 0
 for document in test_df3.text:
  X_test3 = vectorizer3.transform([document])
  out_file3.write(str(clf3.predict(X_test3)[0])+'\n')

 
 out_file3.close()
 
 print "Unigram TF-IDF - Done!"
 
########################
#bigramtfidf.output.txt#
########################
  
 #df = pd.read_csv('imdb_tr.csv')
 vectorizer4 = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
 #corpus = []
 #answers= []

 #for document in df.text:
 # corpus.append(document)

 #for polar in df.polarity:
 # answers.append(polar)
 
 
 X4 = vectorizer4.fit_transform(corpus)
 y4 = answers

 #print len(vectorizer.get_feature_names())


 
 SGD4 = SGDClassifier(loss="hinge", penalty="l2")
 parameters = {'learning_rate': ['optimal', 'constant', 'invscaling'], 'eta0':[.2, .25, .3]}
 clf4 = GridSearchCV(SGD4, parameters, cv = 5)
 
 #clf4 = SGDClassifier(loss="hinge", penalty="l2", learning_rate = 'invscaling', eta0 = .225)
 clf4.fit(X4, y4)

 #test = vectorizer4.transform([df.text[1]])

 #print clf4.predict(test)
 #print df.polarity[1]
 
 
 #print clf4.score(X4,y4)
 #print clf4.best_score_
 #print clf4.best_params_
 
 #Testing
 test_df4 = pd.read_csv(test_path, encoding='cp850')
 
 
 bigram_answer4 = []
 out_file4 = open('bigramtfidf.output.txt', 'w')
 
 only_once = 0
 for document in test_df4.text:
  X_test4 = vectorizer4.transform([document])
  out_file4.write(str(clf4.predict(X_test4)[0])+'\n')

 
 out_file4.close()
 
 
     
 pass