import numpy as np
import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.spatial import distance
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
tqdm.pandas()


df = pd.read_csv('P2_Training_Dataset.csv',encoding="ISO-8859-1")

authordeltas = df[ df['author'] == 'DeltaBot' ].index
authordeleted= df[ df['author'] == '[deleted]' ].index
textnull=df[df['text'] == 'null'].index
# Delete these row indexes from dataFrame
df.drop(authordeltas , inplace=True)
df.drop(authordeleted, inplace=True)
df.drop(textnull, inplace=True)
list_text=list(df['text'])
list_delta=list(df['delta'])
list_thread_id=list(df['thread_id'])



def preprocess_data(list_text):
    comment_length=list()
    comment_score=list()
    comment_hedges=list()
    analyzer=SentimentIntensityAnalyzer()
    hedges_list = []
    with open('hedges.txt', "r") as f:
        for line in f:
            hedges_list.extend(line.split())
            
    cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    for i in range(len(list_text)):
        list_text[i]=str(list_text[i]).encode("ascii", errors="ignore").decode()
        list_text[i]=re.sub(cleanr, '', list_text[i])
        if(list_text[i].rfind("EDIT")):
            list_text[i]=list_text[i][:list_text[i].rfind("EDIT")]
        comment_length.append(len(list_text[i]))
        score=analyzer.polarity_scores(list_text[i])
        comment_score.append(score['compound'])
    for i in list_text:
        c=0
        for j in hedges_list:
            if(j in i):
                c+=1
        comment_hedges.append(c)  
    return list_text,comment_length,comment_score,comment_hedges




lt,c_len,c_sco,c_hed=preprocess_data(list_text)
np.save("score",c_sco)

#c_sco=np.load('score_train')

def getthreads(list_thread_id):
    
    td_shift=[0]
    for i in range(len(list_thread_id)):
        td_shift.append(list_thread_id[i])


    list_thread_id.append(0)

    tflist=[]
    for i in range(len(list_thread_id)):
        if(list_thread_id[i]==td_shift[i]):
            tflist.append("false")
        else:
            tflist.append("true")
        
    tdfinal=[]
    for e,v in enumerate(tflist):
        if v=="true":
            ind=e
            tdfinal.append(ind)
        else:
            tdfinal.append(ind)

    return tdfinal


tdf=getthreads(list_thread_id)



def calc_similarity(lt,tdfinal):
    try:
        vect=CountVectorizer()
        combine=list()
        combine.append(lt)
        combine.append(tdfinal)
        conv_vec=vect.fit_transform(combine)
        v1=conv_vec.toarray()
        a=v1[0].tolist()
        b=v1[1].tolist()
        cos_dist=distance.cosine(a,b)
        return cos_dist
    except:
        return 0
    

list_op=[]
for x,y in zip(list_text,tdf):
    list_op.append(list_text[y])        
df['optext']=list_op
df['text']=list_text

df['cos']=df[['text','optext']].progress_apply(lambda y:calc_similarity(y[0],y[1]),axis=1)

np.save("sim_train",df['cos'])
c_sim=df['cos']

feature_df = pd.DataFrame(
    {'length': c_len,
     'hedge count': c_hed,
     'similarity': c_sim,
     'sentiment': c_sco
    })
    
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(feature_df,list_delta) 

y_pred=neigh.predict(feature_df)
pos_cases=(y_pred==list_delta).sum() 
total=len(list_delta) 
accuracy=pos_cases*100/total 
print("accuracy with respect to training set::",accuracy)



df_test = pd.read_csv('P2_Testing_Dataset.csv',encoding="ISO-8859-1")

authordeltas = df_test[ df_test['author'] == 'DeltaBot' ].index
authordeleted= df_test[ df_test['author'] == '[deleted]' ].index
textnull=df_test[df_test['text'] == 'null'].index
# Delete these row indexes from dataFrame
df_test.drop(authordeltas , inplace=True)
df_test.drop(authordeleted, inplace=True)
df_test.drop(textnull, inplace=True)
list_text=list(df_test['text'])
list_delta=list(df_test['delta'])
list_thread_id=list(df_test['thread_id'])

lt,c_len,c_sco,c_hed=preprocess_data(list_text)

np.save("score_test",c_sco)

tdf=getthreads(list_thread_id)

list_op=[]
for x,y in zip(lt,tdf):
    list_op.append(lt[y])

df_test['optext']=list_op
df_test['text']=lt

df_test['cos']=df_test[['text','optext']].progress_apply(lambda y:calc_similarity(y[0],y[1]),axis=1)

np.save("sim_test",df_test['cos'])


c_sim=df_test['cos']

feature_df_test = pd.DataFrame(
    {'length': c_len,
     'hedge count': c_hed,
     'similarity': c_sim,
     'sentiment': c_sco
    })
    
y_pred=neigh.predict(feature_df_test)

np.savetxt('prediction_featureSet1.csv',y_pred,delimiter=',')

pos_cases=(y_pred==list_delta).sum()   #no of correct matches
total=len(list_delta)                         
accuracy=pos_cases*100/total      #accuracy on training set
print("accuracy with respect to training set::",accuracy)

f1s=f1_score(list_delta, y_pred, average='macro')
print("F1 score::"+str(f1s)+"\n")

ps=precision_score(list_delta, y_pred, average='macro')
print("Precision score::"+str(ps)+"\n") 

rsc=recall_score(list_delta, y_pred, average='macro')  
print("Recall score::"+str(rsc)+"\n") 

cm=confusion_matrix(list_delta, y_pred)
print(cm)

rc_ac=roc_auc_score(list_delta, y_pred)
print("ROC AUC score::"+str(rc_ac)+"\n") 



    
    