import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
tqdm.pandas()
import textstat



df = pd.read_csv('P2_Training_Dataset.csv',encoding="ISO-8859-1")
values = {'author_score': 0}
df=df.fillna(value=values)



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
list_author_score=list(df['author_score'])


def preprocess_data(list_text):
    comment_length=list()
    comment_hedges=list()
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
    for i in list_text:
        c=0
        for j in hedges_list:
            if(j in i):
                c+=1
        comment_hedges.append(c)  
    return list_text,comment_length,comment_hedges


def get_read_score(lt):
    cread=[]
    for i in range(len(lt)):
        cread.append(textstat.flesch_reading_ease(lt[i]))
    return cread
        

lt,c_len,c_hed=preprocess_data(list_text)
c_sco=np.load('score.npy')
c_read=get_read_score(lt)
np.save('readability',c_read)



c_sim=np.load("sim_train.npy")
c_aut_score=list_author_score

feature_df = pd.DataFrame(
    {'length': c_len,
     'hedge count': c_hed,
     'similarity': c_sim,
     'sentiment': c_sco,
     'authorscore':c_aut_score,
     'readability':c_read
    })
    

from sklearn.ensemble import RandomForestClassifier             
#classifier
clf = RandomForestClassifier(n_estimators=100, max_depth=8,
                              random_state=0)   
clf.fit(feature_df,list_delta)
    
df_test = pd.read_csv('P2_Testing_Dataset.csv',encoding="ISO-8859-1")
values = {'author_score': 0}
df_test=df_test.fillna(value=values)

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
list_author_score=list(df_test['author_score'])

lt,c_len,c_hed=preprocess_data(list_text)

c_sco=np.load('score_test.npy')
c_sim=np.load('sim_test.npy')
c_read=get_read_score(lt)
np.save('readability_test',c_read)
c_aut_score=list_author_score
feature_df_test = pd.DataFrame(
    {'length': c_len,
     'hedge count': c_hed,
     'similarity': c_sim,
     'sentiment': c_sco,
     'authorscore':c_aut_score,
     'readability':c_read
    })
    
#y_pred=neigh.predict(feature_df_test)
y_pred=clf.predict(feature_df_test)

np.savetxt('prediction_featureSet2.csv',y_pred,delimiter=',')

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