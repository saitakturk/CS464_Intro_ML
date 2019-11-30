import numpy as  np 
import pandas as pd
np.warnings.filterwarnings('ignore')
#map for 4.4 and 4.5
print('[#] Program is loading...')
def mle(arr,estimator,pi) : 
    estimator = np.nan_to_num(estimator)
    mul = np.multiply(arr, estimator)
    mul[ mul < -1e200 ] = float('-inf')
    return np.log(pi) + np.sum(mul,axis=1,keepdims=True)
#map for 4.6
def mle_bernoulli(arr,estimator,pi) : 
    return  np.log(pi) +np.log( np.prod((arr * estimator + (1 - arr) * ( 1 -estimator)), axis=1, keepdims=True))


#read files
train_features = pd.read_csv("question-4-train-features.csv", sep=',', header = None)
test_features = pd.read_csv("question-4-test-features.csv", sep=',', header = None )
train_labels = pd.read_csv("question-4-train-labels.csv",header=None, sep='\n')
test_labels = pd.read_csv("question-4-test-labels.csv", header = None, sep = '\n')
vocab = pd.read_csv("question-4-vocab.txt", sep = '\t', header = None)

#get arrays
map_labels = {'negative' : 0, 'neutral' : -1, 'positive' : 1}
train_X = train_features.values.astype(np.int32)
train_y = train_labels.values
train_y = np.vectorize(map_labels.get)(train_y).reshape(-1)
test_X = test_features.values
test_y = test_labels.values
test_y = np.vectorize(map_labels.get)(test_y).reshape(-1)
vocab = vocab.values

#classes
negative_tweets = train_X[:][ train_y == 0]
neutral_tweets  = train_X[:][ train_y == -1 ]
positive_tweets = train_X[:][ train_y == 1 ]

'''4.4'''
#estimator 
estimator = lambda arr : np.log(np.sum(arr, axis=0, keepdims=True) / np.sum(arr))

#prior
pi_neg = negative_tweets.shape[0] / train_X.shape[0]
pi_net = neutral_tweets.shape[0]  / train_X.shape[0]
pi_pos = positive_tweets.shape[0] / train_X.shape[0]

#estimator 
estimator_neg = estimator(negative_tweets)
estimator_net = estimator(neutral_tweets)
estimator_pos = estimator(positive_tweets)

#results 
neg = mle(test_X, estimator_neg, pi_neg)
net = mle(test_X, estimator_net, pi_net)
pos = mle(test_X, estimator_pos, pi_pos)
predictions = np.hstack([net,neg,pos])
max_predictions =np.argmax(predictions, axis=1) -1 

#accuracy
acc = max_predictions == test_y

print("4.4 Result : ", len(acc[ acc == True])/len(acc))


'''4.5'''
#estimator function
estimator_with_prior = lambda arr : np.log((np.sum(arr, axis=0, keepdims=True)+1) / (np.sum(arr)+ 1 * arr.shape[1]))
#estimators
p_est_neg = estimator_with_prior(negative_tweets)
p_est_net = estimator_with_prior(neutral_tweets)
p_est_pos = estimator_with_prior(positive_tweets)

#results
p_neg = mle(test_X, p_est_neg, pi_neg)
p_net = mle(test_X, p_est_net, pi_net)
p_pos = mle(test_X, p_est_pos, pi_pos)
predictions = np.hstack([p_net,p_neg,p_pos])
max_predictions =np.argmax(predictions, axis=1) -1 

#accuracy
acc = max_predictions == test_y

print("4.5 Result : ", len(acc[ acc == True])/len(acc))

'''4.6'''
#copy and make 1 every values that are not zero
test_X_ber = test_X.copy()
test_X_ber[ test_X_ber != 0 ] = 1
neg_tw = negative_tweets.copy()
net_tw = neutral_tweets.copy()
pos_tw = positive_tweets.copy()
neg_tw[ neg_tw  != 0 ] = 1 
net_tw[ net_tw != 0 ] = 1
pos_tw[ pos_tw != 0 ] = 1
ber_estimator = lambda arr : np.sum(arr, axis = 0, keepdims = True) / arr.shape[0]

#estimators
neg_ber_est = ber_estimator(neg_tw)
net_ber_est = ber_estimator(net_tw)
pos_ber_est = ber_estimator(pos_tw)


#results
ber_neg = mle_bernoulli(test_X_ber, neg_ber_est, pi_neg)
ber_net = mle_bernoulli(test_X_ber, net_ber_est, pi_net)
ber_pos = mle_bernoulli(test_X_ber, pos_ber_est, pi_pos)

predictions = np.hstack([ber_net,ber_neg,ber_pos])
max_predictions =np.argmax(predictions, axis=1) -1

#accuracy
acc = max_predictions == test_y
print("4.6 Result : ", len(acc[ acc == True])/len(acc))


'''4.7'''
top_words= lambda tweet_class : vocab[np.argsort(np.sum(tweet_class, axis=0))[-20:]][:,0]
negative_tweets_top_words = top_words(negative_tweets)
neutral_tweets_top_words = top_words(neutral_tweets)
positive_tweets_top_words = top_words(positive_tweets)
print('\n4.7\n')
print('\nNegative top words : \n')
print(negative_tweets_top_words)
print('\nNeutral top words : \n')
print(neutral_tweets_top_words)
print('\nPositive top words : \n')
print(positive_tweets_top_words)

