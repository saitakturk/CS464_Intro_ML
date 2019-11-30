import numpy as np
import pandas as pd 
import cv2
import matplotlib.pyplot as plt 
import os
#from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
from sklearn.decomposition import PCA
from itertools import chain
import itertools
from sklearn.svm import LinearSVC, SVC
################################################Q1####################################################
def read_files(abs_path = './lfwdataset/'):
    data = {}
    img_paths = os.listdir(abs_path)
    data['imgs'] = np.array([ cv2.imread(abs_path + os.sep + path,-1).astype(np.float64).reshape(-1) for path in img_paths if path.endswith('.pgm') ])
    data['train_X'] = pd.read_csv('question-2-train-features.csv', header= None, sep = ',').values
    data['train_y'] = pd.read_csv('question-2-train-labels.csv', header = None, sep = ',').values
    data['test_X'] = pd.read_csv("question-2-test-features.csv", header = None, sep =',').values
    data['test_y'] = pd.read_csv('question-2-test-labels.csv', header = None, sep=',').values
    return data

def pca_apply(imgs, k):
    #my implementation
    # k = 4096
    # imgs_norm = imgs  - np.mean(imgs, axis=1).reshape(-1,1)
    # cov_matrix = np.dot(imgs_norm.T, imgs_norm)
    # def eig_val_and_vec(cov_matrix, k):
    #     eig_vals, eig_vecs = largest_eigsh(cov_matrix, k, which='LM')
    #     return eig_vals[::-1],eig_vecs[:,::-1]
    # def recons_img(k,imgs_norm, imgs, eig_vecs):
    #     return np.dot(np.dot(imgs_norm,eig_vecs[:,:k]),eig_vecs[:,:k].T) + np.mean(imgs, axis=1).reshape(-1,1)
    # eig_vals, eig_vecs = eig_val_and_vec(cov_matrix, k)
    # imgsx = recons_img(4096,imgs_norm, imgs, eig_vecs)
    pca_face = PCA(k)
    pcax = pca_face.fit_transform(imgs) 
    return pca_face, pca_face.components_, pcax


def recons_img(eig_vecs, mean, imgs):
    #np.dot(np.dot(imgs_norm.T,eig_vecs[:,:k]),eig_vecs[:,:k].T) + np.mean(imgs, axis=0).reshape(-1,1)
    reduced_imgs = np.dot(imgs - mean, eig_vecs.T)
    return np.dot(reduced_imgs, eig_vecs) + mean





def q1():
    data = read_files()
    imgs = data['imgs']
    pca_face, eig_vecs, pca_result = pca_apply(imgs, 4096)
    imgsx = recons_img(pca_face.components_, pca_face.mean_, imgs)
    ##1.2
    print('[#]1.2 is loading...')
    exp_var = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    pca_exp_var = [np.sum(pca_face.explained_variance_ratio_[:x]) for x in exp_var]
    plt.plot(exp_var, pca_exp_var )
    plt.title("Explained Variance Ratio")
    plt.ylabel("Ratio")
    plt.xlabel("Eigen Values ")
    plt.show()

    ##1.3
    print('[#]1.3 is loading...')
    fig=plt.figure(figsize=(10,10))
    columns = 10
    rows = 5
    #first row
    titles = ['Aaron Eckhart', 'Aaron Guiel', 'Aaron Patterson', 'Aaron Peirsol','Aaron Peirsol','Aaron Peirsol','Aaron Peirsol',
            'Aaron Pena','Aaron Sorkin', 'Aaron Sorkin']
    for i in range(1, 11):
        fig.add_subplot(rows, columns, i)
    
        plt.imshow(imgs[i-1].reshape(64,64), cmap=plt.get_cmap('gray'))
        plt.title(titles[i-1])
    
            
    for i in range(1, 11):
        fig.add_subplot(rows, columns, i+10)
        plt.imshow(eig_vecs[i-1,:].reshape(64,64), cmap = plt.get_cmap('gray'))
        plt.title('{}. Eigen Face'.format(i))


    pca_face, eig_vecs, pca_result = pca_apply(imgs, 32)
    imgsx = recons_img(pca_face.components_, pca_face.mean_, imgs)

    for i in range(1, 11):
        fig.add_subplot(rows, columns, i+20)

        plt.imshow(imgsx[i-1].reshape(64,64), cmap=plt.get_cmap('gray'))
        plt.title("32 Eigenfaces\n{}".format(titles[i-1]))


            
    pca_face, eig_vecs, pca_result = pca_apply(imgs, 128)
    imgsx = recons_img(pca_face.components_, pca_face.mean_, imgs)

    for i in range(1, 11):
        fig.add_subplot(rows, columns, i+30)
        
        plt.imshow(imgsx[i-1].reshape(64,64), cmap=plt.get_cmap('gray'))
        plt.title("128 Eigenfaces\n{}".format(titles[i-1]))
    
            

    pca_face, eig_vecs, pca_result = pca_apply(imgs, 512)
    imgsx = recons_img(pca_face.components_, pca_face.mean_, imgs)

    for i in range(1, 11):
        fig.add_subplot(rows, columns, i+40)
    
        plt.imshow(imgsx[i-1].reshape(64,64), cmap=plt.get_cmap('gray'))
        plt.title("512 Eigenfaces\n{}".format(titles[i-1]))

    plt.rcParams['axes.labelweight'] = 'bold'
    plt.show()
    del data
    print('[#]1.4 is loading...')
    print('1.4 : 14.4%')
################################################Q1####################################################

################################################Q2####################################################

#linear regression
def weight_initializer(weight_size = (9,1)):
    return np.random.normal(size=weight_size, loc=0,scale=0.01)

def mse( y,y_):
    return np.mean(np.square(y - y_ )) 

def add_one_column(X):
    return np.hstack((X, np.ones((X.shape[0],1))))
def weight_linear(X,y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


def q2():
    data = read_files()
    print('[#]2.2 is loading...')
    print("Rank : ", np.linalg.matrix_rank(np.dot(data['train_X'].T, data['train_X'])))
    print('[#]2.3 is loading...')
    b = 0
    W = weight_initializer()
    X = data['train_X'].copy()
    y = data['train_y']
    test_X = data['test_X'].copy()
    test_y = data['test_y']
    X = add_one_column(X)
    W = weight_linear(X,y)
    print("Weights : ",W)
    print()
    y_ = np.hstack((test_X, np.ones((test_X.shape[0],1)))).dot(W)
    y_tr = X.dot(W)
    print("Test mse : ",mse(test_y, y_))
    print("Train mse : ",mse(y,y_tr))
    print('[#]2.4 is loading...')
    hum = np.hstack([data['train_X'][:,6], data['test_X'][:,6]]).reshape(-1,1)
    rentals = np.vstack([data['train_y'], data['test_y']])
    plt.scatter(hum, rentals,marker='*',s=0.2,c= 'green', label = 'Data Points')
    plt.ylabel("Rental Bikes")
    plt.xlabel("Normalized Humidity")
    plt.title("Normalized Humidity vs Rental Bikes")
    plt.legend()
    plt.show()
    print('[#]2.5 is loading...')
    W = weight_linear(add_one_column(hum), rentals)
    plt.scatter(hum, rentals,marker='*',s=0.2,c= 'green', label = 'Data Points')
    plt.ylabel("Rental Bikes")
    plt.xlabel("Normalized Humidity")
    plt.title("Normalized Humidity vs Rental Bikes")
    rangex = np.linspace(0.0,1.0,1000)
    y = W[0] * rangex + W[1]
    plt.plot(rangex,y,c='red', label='Regression Line' )
    plt.legend()
    plt.show()
    del data
################################################Q2####################################################
################################################Q3####################################################
def macro_metrics(cm):
    TP1 = cm[0,0]
    TN1 = cm[1,1]
    FP1 = cm[0,1]
    FN1 = cm[1,0]
    
    TP2 = cm[1,1]
    TN2 = cm[0,0]
    FP2 = cm[1,0]
    FN2 = cm[0,1]
    
    prec1 = TP1 / (TP1 + FP1)
    prec2 = TP2 / (TP2 + FP2)
    
    rec1 = TP1 / (TP1 + FN1)
    rec2 = TP2 / (TP2 + FN2)
    
    npv1 = TN1 / ( TN1 + FN1)
    npv2 = TN2 / ( TN2 + FN2)
    
    fpr1 = FP1 / (FP1 + TN1)
    fpr2 = FP2 / (FP2 + TN2)
    
    fdr1 = FP1 / (FP1 + TP1)
    fdr2 = FP2 / (FP2 + TP2)
    def f_beta_score(beta, precision, recall) : return (beta*beta + 1)*precision*recall / (beta*beta*precision + recall)
    f11 = f_beta_score(1, prec1, rec1)
    f12 = f_beta_score(1, prec2, rec2 )
    
    f21 = f_beta_score(2, prec1, rec1)
    f22 = f_beta_score(2, prec2, rec2 )
   
    metric = {}
    metric['prec'] = (prec1 + prec2)/2
    metric['rec'] = (rec1 + rec2 )/2
    metric['npv'] = (npv1 + npv2)/2
    metric['fpr'] = (fpr1 + fpr2)/2
    metric['fdr'] = (fdr1 + fdr2)/2
    metric['f1'] = (f11 + f12 )/2
    metric['f2'] = (f21 + f22 )/2
    
    return metric

def micro_metrics(cm):
    TP1 = cm[0,0]
    TN1 = cm[1,1]
    FP1 = cm[0,1]
    FN1 = cm[1,0]
    
    TP2 = cm[1,1]
    TN2 = cm[0,0]
    FP2 = cm[1,0]
    FN2 = cm[0,1]
    
    prec = (TP1 + TP2) / (TP1 + FP1 + TP2 + FP2)
   
    
    rec = ( TP1 + TP2 ) / (TP1 + FN1 + TP2 + FN2)
 
    
    npv = ( TN1 + TN2 )  / ( TN1 + FN1 + TN2 + FN2 )
    
    
    fpr = (FP1 + FP2 )  / (FP1 + TN1 + FP2 + TN2)
    
    
    fdr = ( FP1 + FP2) / (FP1 + TP1 + FP2 + TP2)
  
    def f_beta_score(beta, precision, recall) : return (beta*beta + 1)*precision*recall / (beta*beta*precision + recall)
    f1 = f_beta_score(1, prec, rec)
  
    
    f2 = f_beta_score(2, prec, rec)
   
   
    metric = {}
    metric['prec'] = prec
    metric['rec'] = rec
    metric['npv'] = npv
    metric['fpr'] = fpr
    metric['fdr'] = fdr
    metric['f1'] =  f1
    metric['f2'] =  f2
    
    return metric


def conf_matrix_(test_X, weight, bias, test_y):
    z = forward(test_X, weight, bias)
    pred, loss = sigmoid_with_cross_entropy(z,test_y)
    pred[ pred > 0.5 ] = 3
    pred[ pred <= 0.5 ] = 0
    a = pred - test_y
    conf_matrix = np.zeros((2,2)).astype(np.int32)
    conf_matrix[0, 0] = len(a[ a == 2 ])
    conf_matrix[1, 0] = len(a[ a == 3 ])
    conf_matrix[0, 1] = len(a[ a == -1 ])
    conf_matrix[1, 1] = len(a[ a == 0 ])
    return conf_matrix

def plot_conf_matrix(cm, classes, title='Confusion matrix',cmap=plt.cm.Blues):
    '''
    Plots confusion matrix, heavily inspired from scikit.learn website
    '''
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j],  'd'),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    #plt.savefig("conf_batch_mini.png", layout = 'tight', dpi=300)
    plt.show()

def forward(X, w, b):
    
    return np.dot(X, w) + b 

def sigmoid_with_cross_entropy(z, y,derive = False):
    
    #z is sigmoid result
    if derive:
        return y - z
    
    sigmoid  = 1 / ( 1 + np.exp(-z))
    return sigmoid, np.sum(np.multiply(y,z) - np.log(1+np.exp(z) + np.exp(-32))) / y.shape[0]

def backward(da,X,w,b, lr_rate = 1e-3):
    
    db = np.sum(da) / da.shape[0]
    dw = np.dot(X.T,da) / da.shape[0]
    
    w = w  + lr_rate * dw
    b = b + lr_rate * db
    
    return w,b

def random_shuffle(X,y):
    sec_idx = np.random.permutation(y.shape[0])
    return  X[sec_idx,:], y[sec_idx,:] 

def q3():
    data = read_files()
    mean_label = np.vstack([data['train_y'],data['test_y']]).mean()

    train_X = data['train_X'].copy()
    train_y = data['train_y'].copy()
    test_X = data['test_X'].copy()
    test_y = data['test_y'].copy()

    train_y[ train_y < mean_label ] = 0
    train_y[ train_y >= mean_label ] = 1

    test_y[ test_y < mean_label ] = 0
    test_y[ test_y >= mean_label ] = 1
    print('[#]3.1 is loading...')
    weight = np.zeros((8,1))
    bias = np.zeros(1)
    X,y = train_X, train_y
    #X,y = random_shuffle(train_X, train_y)
    for i in range(1000):
        #X,y = random_shuffle(X, y)
        z = forward(X, weight, bias)
        a, loss = sigmoid_with_cross_entropy(z, y)
        if ( (i+1) % 100 == 0 ):
            print("Epoch {} --> Loss :".format(i+1), loss)
        da = sigmoid_with_cross_entropy(a,y,True)
        weight, bias = backward(da, X, weight, bias, lr_rate =   1e-1)

    conf_matrix = conf_matrix_(test_X, weight, bias, test_y )    
    print("Accuracy learning rate 0.1 : ", (conf_matrix[0,0]+conf_matrix[1,1])/ np.sum(conf_matrix))
    plot_conf_matrix(conf_matrix, ["Y=1", 'Y=0'])
    print("Micro Average : \n")
    print(micro_metrics(conf_matrix))
    print("Macro Average : \n")
    print(macro_metrics(conf_matrix))
    print('[#]3.2 is loading...')
    del weight, bias
    weight = np.zeros((8,1))
    bias = np.zeros(1)
    batch_size = 32 
    X,y = train_X, train_y
    for i in range(1000):
        #     X,y = random_shuffle(train_X, train_y)
        empty = []
        for j in chain(range(batch_size,y.shape[0],batch_size),[y.shape[0]]):
            z = forward(X[j-batch_size:j], weight, bias)
            a, loss = sigmoid_with_cross_entropy(z, y[j-batch_size:j])
            empty.append(loss)
            da = sigmoid_with_cross_entropy(a,y[j-batch_size:j],True)
            weight, bias = backward(da, X[j-batch_size:j], weight, bias, lr_rate = 0.1)
        if( (i+1) % 100 == 0):
            print("Epoch {} --> Loss :".format(i+1),np.mean(empty))
    conf_matrix = conf_matrix_(test_X, weight, bias, test_y )    
    print("Accuracy learning rate 0.1 : ", (conf_matrix[0,0]+conf_matrix[1,1])/ np.sum(conf_matrix))
    plot_conf_matrix(conf_matrix, ["Y=1", 'Y=0'])
    print("Micro Average : \n")
    print(micro_metrics(conf_matrix))
    print("Macro Average : \n")
    print(macro_metrics(conf_matrix))

    del data
################################################Q3####################################################
################################################Q4####################################################
def conf_matrix_svm(pred,test_y):

    pred[ pred == 1 ] = 3
    a = pred - test_y
    conf_matrix = np.zeros((2,2)).astype(np.int32)
    conf_matrix[0, 0] = len(a[ a == 2 ])
    conf_matrix[1, 0] = len(a[ a == 3 ])
    conf_matrix[0, 1] = len(a[ a == -1 ])
    conf_matrix[1, 1] = len(a[ a == 0 ])
    return conf_matrix


def q4():
    print('[#]4.1 is loading...')
    data = read_files()
    mean_label = np.vstack([data['train_y'].copy(),data['test_y'].copy()]).mean()

    train_X = data['train_X'].copy()
    train_y = data['train_y'].copy()
    test_X = data['test_X'].copy()
    test_y = data['test_y'].copy()

    train_y[ train_y < mean_label ] = 0
    train_y[ train_y >= mean_label ] = 1

    test_y[ test_y < mean_label ] = 0
    test_y[ test_y >= mean_label ] = 1


    C = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
    datax = {}
    train_X, train_y = random_shuffle(train_X, train_y)
    for j in C:
        print("Testing C For ",j)
        svm = LinearSVC(C=j, random_state=2019)
        split_data = np.split(train_X, 10)
        split_data = np.array(split_data)
        split_label = np.split(train_y, 10)
        split_label = np.array(split_label)
        sum_acc = 0 
        for i in range(10):
            val_data = split_data[i]
            val_label = split_label[i]
            train_data = np.concatenate(split_data[list(set(range(10)) - set({i}))])
            train_label =  np.concatenate(split_label[list(set(range(10)) - set({i}))])

            svm.fit(train_data, train_label.reshape(-1))
            
            sum_acc += svm.score(val_data,val_label.reshape(-1))
            #print(i)
        datax[j] = sum_acc / 10.0 
        
    max_key = max(datax, key=datax.get)
    print("Optimal C is ", max_key)
    svm = LinearSVC(C=max_key,random_state=2019)
    svm.fit(train_X, train_y.reshape(-1))

    print("Test Accuracy : ",svm.score(test_X, test_y.reshape(-1)))

    plt.plot(datax.keys(), datax.values())
    plt.xscale('log')
    plt.title("C vs Accuracy")
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.show()

    pred = svm.predict(test_X).reshape(-1, 1)
    conf_matrix = conf_matrix_svm(pred, test_y)  
    #plot_conf_matrix(conf_matrix, ["Y=1", "Y=0"])

    print("Accuracy learning rate 0.1 : ", (conf_matrix[0,0]+conf_matrix[1,1])/ np.sum(conf_matrix))
    plot_conf_matrix(conf_matrix, ["Y=1", 'Y=0'])
    print("Micro Average : \n")
    print(micro_metrics(conf_matrix))
    print("Macro Average : \n")
    print(macro_metrics(conf_matrix))

    print('[#]4.2 is loading...')
    gamma = [ 2**-4, 2**-3, 2**-2, 2**-1, 2**0, 2**1]
    dataf = {}
    train_X, train_y = random_shuffle(train_X, train_y)
    for j in gamma:
        print("Testing gamma for : ",j)
        svm = SVC(C=1e3, gamma = j , kernel ='rbf',random_state=2019)
        split_data = np.split(train_X, 10)
        split_data = np.array(split_data)
        split_label = np.split(train_y, 10)
        split_label = np.array(split_label)
        sum_acc = 0 
        for i in range(10):
            val_data = split_data[i]
            val_label = split_label[i]
            train_data = np.concatenate(split_data[list(set(range(10)) - set({i}))])
            train_label =  np.concatenate(split_label[list(set(range(10)) - set({i}))])

            svm.fit(train_data, train_label.reshape(-1))
            
            sum_acc += svm.score(val_data,val_label.reshape(-1))
            #print(i)
        dataf[j] = sum_acc / 10.0 

    max_key = max(dataf, key=dataf.get)
    print("Optimal gamma is ", max_key)
    svm = SVC(C=1e3, gamma =max_key, kernel = 'rbf', random_state=2019)
    svm.fit(train_X, train_y.reshape(-1))


    print("Test Accuracy : ",svm.score(test_X, test_y.reshape(-1)))



    pred = svm.predict(test_X).reshape(-1, 1)
    conf_matrix = conf_matrix_svm(pred, test_y)  
    #plot_conf_matrix(conf_matrix, ["Y=1", "Y=0"])

    print("Accuracy learning rate 0.1 : ", (conf_matrix[0,0]+conf_matrix[1,1])/ np.sum(conf_matrix))
    plot_conf_matrix(conf_matrix, ["Y=1", 'Y=0'])
    print("Micro Average : \n")
    print(micro_metrics(conf_matrix))
    print("Macro Average : \n")
    print(macro_metrics(conf_matrix))

    del data
################################################Q4####################################################

#q1()
#q2()
#q3()
#q4()
import sys
if len(sys.argv) > 1:
    question = sys.argv[1]
else:
    question = 'run_all'


def sait_akturk_21501734_hw2(question):
    if question == 'run_all':
        print("Running All...")
        q1()
        q2()
        q3()
        q4()
    if question == '1' :
        print (question)
        q1()
    elif question == '2' :
        print (question)
        q2()
    elif question == '3' :
        print (question)
        q3()
    elif question == '4' :
        print (question)
        q4()
sait_akturk_21501734_hw2(question)