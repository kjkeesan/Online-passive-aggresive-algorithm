import pandas as pd
import numpy as np

data=pd.read_csv("wisconsin-data.csv")
x_id=data.iloc[:,0]
x_test_id=x_id.iloc[466:]
data['prediction'].replace([2,4],[-1,1],inplace=True)

data.drop('id',axis=1,inplace=True)


y=data.iloc[:,-1]
#print(y)
x=data.iloc[:,0:9] 

x_train=x.iloc[0:466] 

y_train=y.iloc[0:466] 

x_test=x.iloc[466:]

y_test=y.iloc[466:]

#Here we have to look 3 update of weight. PA1 PA2 PA3 are three algorithms
#PA1==1 PA2==2 PA3==3
algo=[1,2,3]
for s in algo:
    
    iteration=[1,2,10]
    for i in iteration:
        W=np.zeros(9)
        C=1
        
        for r in range(i):
            for p in range(x_train.shape[0]):
                x_tr=x_train.iloc[p,:]
                y_tr=y_train.iloc[p]
            
                loss=max(0,1-(y_tr*np.dot(W.T,x_tr)))
                
                if(s==1):
                    tau=loss/ (np.power(np.linalg.norm(x_tr,ord=2),2))
                   
                if(s==2):
                    tau=min(C,loss /(np.power(np.linalg.norm(x_tr,ord=2),2)))
                    
                if(s==3):
                    tau = loss / (np.power(np.linalg.norm(x_tr,ord=2),2))+(1/(2*C))
                    
                W = W + tau*y_tr*x_tr
                



        #print(i)
        #print(W)


            #training_accuracy
        correct_train_count=0
        for c in range(x_train.shape[0]):
            
            x_tr=x_train.iloc[c,:]
            
            y_tr=y_train.iloc[c]
            
            y_pred=np.sign(np.dot(W.T,x_tr))
            
            if(y_pred*y_tr==1):
                
                correct_train_count+=1
                
            train_accuracy=(correct_train_count/x_train.shape[0])*100

            #Testing_accuracy
            
        print("Algorithm", s)
        print(i)
        print("Training accuracy of the dataset: " , train_accuracy)
        
        correct_test_count=0
        
        for c in range(x_test.shape[0]):
            
            x_tes=x_test.iloc[c,:]
            
            y_tes=y_test.iloc[c]
            
            y_pred=np.sign(np.dot(W.T, x_tes))
            
            #print(x_test_id.iloc[c],"   ",y_test.iloc[c]," ",int(y_pred))
            if(y_pred*y_tes==1):
                
                correct_test_count+=1
                
            test_accuracy=(correct_test_count/x_test.shape[0])*100
            
        print("Algorithm", s)
        print(i)
        print("Testing accuracy of the dataset: " , test_accuracy)

       


        


        
            
                           
            
            



            
                           

                



