import numpy as np
import pandas as pd
import re
from tqdm import tqdm
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



        

lt,c_len,c_hed=preprocess_data(list_text)
c_sco=np.load('score.npy')
c_sim=np.load('sim_train.npy')
c_read=np.load('readability.npy')






feature_df = pd.DataFrame(
    {'similarity': c_sim,
     'sentiment': c_sco,
     'readability':c_read
    })
    
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(feature_df,list_delta) 

     
from sklearn.ensemble import RandomForestClassifier             
#classifier
clf = RandomForestClassifier(n_estimators=100, max_depth=8,
                              random_state=0)   
clf.fit(feature_df,list_delta)



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

c_sco=np.load('score_test.npy')
c_sim=np.load('sim_test.npy')
c_read=np.load('readability_test.npy')



feature_df_test = pd.DataFrame(
    {'similarity': c_sim,
     'sentiment': c_sco,
     'readability':c_read
    })


y_pred=clf.predict(feature_df_test)
np.savetxt('prediction_featureSet3.csv',y_pred,delimiter=',')


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
