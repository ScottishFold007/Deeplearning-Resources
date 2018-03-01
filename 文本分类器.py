#导入必要的库
from sklearn.datasets import load_files #导入数据加载库
from sklearn.cross_validation import train_test_split #分割训练集和测试集

path=r'C:\Users\gaochangkuan\Desktop\xiaoshuo' #训练语料和测试语料的路径
twenty_train = load_files(container_path=path,decode_error='ignore',encoding='utf-8') #加载语料


#分割训练集和测试集，训练集和测试机的比值为75%：25%
X_train,X_test,y_train,y_test=train_test_split(twenty_train.data,twenty_train.target)



#导入jieba分词库，对文本进行分词
import jieba  

#自定义分词器
jiebatonkenizer=lambda x:jieba.cut(x)

#设置停用词
stwlist=[line.strip() for line in open(r'C:\Users\gaochangkuan\Desktop\extra_dict\停用词汇总.txt','r',encoding='utf-8').readlines()]

from sklearn.feature_extraction.text import CountVectorizer
#其中重点关注的两个参数tokenizer和stop_words，需要用到前面自定义的分词（jiebatonkenizer）和停用词词典（stwlist）
count_vect = CountVectorizer(tokenizer=jiebatonkenizer,stop_words=stwlist) #从文本中提取TF（词频）特征
X_train_counts = count_vect.fit_transform(X_train) #将文本转为词频矩阵


#提取特征，利用TfidfVectorizer，把训练文本转换成TF特征的矩阵
from sklearn.feature_extraction.text import TfidfTransformer 
#TfidfTransformer用于统计X_train_counts中每个词语的TF值
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)


#TfidfTransformer用于统计X_train_counts中每个词语的TF-IDF值，把训练文本转换成TF-IDF特征的矩阵
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts) 


#导入朴素贝叶斯分类器 
from sklearn.naive_bayes import MultinomialNB #导入MultinomialNB多项式模型

clf = MultinomialNB().fit(X_train_tfidf, y_train) #用多项式模型训练文本分类器


#验证模型准确性
from sklearn.pipeline import Pipeline #导入pipeline
text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()), ('clf', MultinomialNB()), ]) #为pipeline设置不同的参数时，组合几个可以一起交叉验证的步骤
text_clf = text_clf.fit(X_train, y_train) #用设置好的pipeline来fit训练数据

#评价训练好的文本分类器
import numpy as np
docs_test = X_test
predicted = text_clf.predict(docs_test) #设置好对测试集的预测
print(np.mean(predicted == y_test)) #文本训练器的准确度



#运用支持向量机进行分类训练文本分类器，并检验其准确性
from sklearn.linear_model import SGDClassifier #导入线性核函数的模型中的随机梯度下降分类器模块
text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', 
penalty='l2',alpha=1e-3, n_iter=5, random_state=50))]) #将前面的步骤并入到pipeline中
_= text_clf.fit(X_train,y_train) #用设置好的pipeline来fit训练数据
predicted = text_clf.predict(X_test) #设置好对测试集的预测
print(np.mean(predicted == y_test)) #文本训练器的准确度


#用语句测试分类器效果，ps:最好是用文档来测，这样文本量够大，会更准确些
docs_new = ['在大胡子肌肉男胳膊的通知下，后面两个斜眼男子此时慢慢地转过身来围住了绝对美女并且一步步的朝着绝对美女开始逼近。这个混蛋竟然如此胆大，竟然敢在老子面前如此对待一个绝对美女，不是找死是做什么？”眼见三人再次欺近绝对美女，冷傲天却也慢慢的朝着这大胡子肌肉男身边靠近着。',
            "是布鲁诺检察官和萨姆巡官吗？麻烦这边请。”这位圆滚滚的老佣人又来了个仿佛柔软体操的行礼，开心地走在前头，引领这两人走入了十六世纪。眼前，是一座广阔到令人肃然一惊的庄园式贵族大厅，天花板上巨大的横梁交错纵横，盔甲闪亮宛如传统的武士，独自守护着室内悬挂的各种古老的饰物和图书。在最远的那面墙上，气势之雄伟诡异，胜过北欧神话里供奉着阵亡将士英灵的瓦尔哈拉神殿一筹，一幅巨型的喜剧面具眯着眼笑得人毛骨悚然。",
            "她轻轻推开门，“校长！”刚叫了一声，发现里面除了校长还有一个人。那是一个个子高高的男学生，背对着门，正在聆听校长训示。 ",
            '师父，您要托弟子们何事？”韦庄安抚着楚雀，抬首问向尊师。在师弟们面前，他竭力维持长兄的威严，忍下与待他如亲父的尊师死别欲哭的情绪。',
            '中国使用特种部队的历史很久远，红军时代就有精锐“手枪队”；抗日战争中“敌后武工队”大显神威；朝鲜战争时期，中国特种部队曾炸毁美军重要桥梁，破坏美军整个战役布势，最有名的是中国特种部队奇袭南韩最精锐的首都师白虎团团部的行动，为中国粉碎白虎团作出了决定性贡献。',
            "纱帘下人如玉，雪色清光耀亮双眼，她的呼吸拂在耳侧，轻浅而幽香，带着隐忍与节制的欢娱。帘幕里逶迤唇齿，无人知这一刻幸福来得如此缠绵，瓷枕上黑发交缠，但愿这一生永远撕脱不开。"]
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))

