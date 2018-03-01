
###预设置：导入必要的库###
from sklearn.datasets import load_files #导入数据加载库
from sklearn.cross_validation import train_test_split #分割训练集和测试集
import jieba 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline #导入pipeline
import numpy as np
from sklearn.linear_model import SGDClassifier #导入线性核函数的模型中的随机梯度下降分类器模块


###第一步：加载训练语料，并分割语料为训练集和测试集###

path=r'C:\Users\gaochangkuan\Desktop\分类' #训练语料和测试语料的路径
twenty_train = load_files(container_path=path,decode_error='ignore',encoding='utf-8') #加载语料

#分割训练集和测试集，训练集和测试机的比值为75%：25%
X_train,X_test,y_train,y_test=train_test_split(twenty_train.data,twenty_train.target)



###第二步：进行文本预处的提前设置：文本分词和去停用词###

#导入jieba分词库，对文本进行分词
jiebatonkenizer=lambda x:jieba.cut(x) #自定义中文文本分词器

#设置停用词
stwlist=[line.strip() for line in open(r'C:\Users\gaochangkuan\Desktop\extra_dict\停用词汇总.txt','r',encoding='utf-8').readlines()]


###第三步：进行文本特征提取###

count_vect = CountVectorizer(tokenizer=jiebatonkenizer,stop_words=stwlist) #从文本中提取TF（词频）特征，其中重点关注的两个参数tokenizer和stop_words，需要用到前面自定义的分词（jiebatonkenizer）和停用词词典（stwlist）
X_train_counts = count_vect.fit_transform(X_train) #将文本转为词频矩阵
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts) #TfidfTransformer用于统计X_train_counts中每个词语的TF值，把训练文本转换成TF特征的矩阵
X_train_tf = tf_transformer.transform(X_train_counts)

tfidf_transformer = TfidfTransformer()#TfidfTransformer用于统计X_train_counts中每个词语的TF-IDF值
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)  #把训练文本转换成TF-IDF特征的矩阵


###第四步：进行文本分类器的训练，并验证模型的准确性###

clf = MultinomialNB().fit(X_train_tfidf, y_train) #导入朴素贝叶斯分类器中的MultinomialNB多项式模型，用多项式模型训练文本分类器


####第四步：model持久化，保存到本地，注意有3个文件####
from sklearn.externals import joblib
joblib.dump(clf,r'C:\Users\gaochangkuan\Desktop\clf.pkl',compress=3) #保存分类器
joblib.dump(count_vect.vocabulary_,r'C:\Users\gaochangkuan\Desktop\count_vect.vocabulary_.pkl',compress=3) #保存文本特征向量，即X
joblib.dump(twenty_train.target_names,r'C:\Users\gaochangkuan\Desktop\twenty_train.target_names.pkl',compress=3) #保存标签数据，即Y


####第五步：使用模型####
#如果是直接调用模型的话，需要一次性加载所需的库
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
import jieba 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pandas as pd

jiebatonkenizer=lambda x:jieba.cut(x,cut_all=True) #自定义分词器

stwlist=[line.strip() for line in open(r'C:\Users\gaochangkuan\Desktop\extra_dict\停用词汇总.txt','r',encoding='utf-8').readlines()] #设置停用词

#调用之前保存报本地的模型，共计3个
clf=joblib.load(r'C:\Users\gaochangkuan\Desktop\clf.pkl')
vocabulary_to_load=joblib.load(r'C:\Users\gaochangkuan\Desktop\count_vect.vocabulary_.pkl')
target_names=joblib.load(r'C:\Users\gaochangkuan\Desktop\twenty_train.target_names.pkl')

loaded_vectorizer=CountVectorizer(tokenizer=jiebatonkenizer,stop_words=stwlist,vocabulary=vocabulary_to_load ) #载入特征提取器，注意模型的本地调用
tfidf_transformer=TfidfTransformer()
 
 #载入新文档
s=open(r'C:\Users\gaochangkuan\Desktop\xiaoshuo\军事小说\filename_front.part18168.txt',encoding='utf-8').readlines() #读取整本小说

X_new_counts=loaded_vectorizer.transform(s) #使该小说的文本数据向量化
X_new_tfidf=tfidf_transformer.fit_transform(X_new_counts) #进一步转化为tf-idf特征矩阵
predicted=clf.predict(X_new_tfidf) #预测结果

#打印对应语句及其预测结果        
for doc,category in zip(s,predicted):
    print('%r=> %s '% (doc,target_names[category]))




