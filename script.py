import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD 
    means = np.zeros((2,5))
    count = 0
    classes = np.zeros(5)
    for i in y:
        for j in range(2):
            means[j][int(i) - 1] = means[j][int(i) - 1] + X[count][j]
        count = count + 1
        classes[int(i) - 1] = classes[int(i) - 1] + 1
    
    for i in range(5):
        for j in range(2):
            means[j][i] = means[j][i] / classes[i]
    
    #print(means)
    return means,np.asarray(np.cov(np.transpose(X)))

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    means = np.zeros((2,5))
    classes = np.zeros(5)
    list1 = np.zeros((2,31))
    list2 = np.zeros((2,39))
    list3 = np.zeros((2,29))
    list4 = np.zeros((2,26))
    list5 = np.zeros((2,25))
    count = 0
    count_1 = 0
    count_2 = 0
    count_3 = 0
    count_4 = 0
    count_5 = 0
    for i in y:
        for j in range(2):
            means[j][int(i) - 1] = means[j][int(i) - 1] + X[count][j]
        if(i == 1):
            list1[0][count_1] = X[count][0]
            list1[1][count_1] = X[count][1]
            count_1 = count_1 + 1
            
        if(i == 2):
            list2[0][count_2] = X[count][0]
            list2[1][count_2] = X[count][1]
            count_2 = count_2 + 1
            
        if(i == 3):
            list3[0][count_3] = X[count][0]
            list3[1][count_3] = X[count][1]
            count_3 = count_3 + 1
            
        if(i == 4):
            list4[0][count_4] = X[count][0]
            list4[1][count_4] = X[count][1]
            count_4 = count_4 + 1
            
        if(i == 5):
            list5[0][count_5] = X[count][0]
            list5[1][count_5] = X[count][1]
            count_5 = count_5 + 1
                
        count = count + 1
        classes[int(i) - 1] = classes[int(i) - 1] + 1
    
    for i in range(5):
        for j in range(2):
            means[j][i] = means[j][i] / classes[i]
    
    count = 0
    covariance_sum = np.zeros((2,5))    
    for i in y:
        for j in range(2):
            covariance_sum[j][int(i) - 1] = covariance_sum[j][int(i) - 1] + ((X[count][j] - means[j][int(i) - 1]) * (X[count][j] - means[j][int(i) - 1])) 
        count = count + 1
    
    for i in range(5):
        for j in range(2):
            covariance_sum[j][i] = covariance_sum[j][i] / classes[i] 
    
    covmats = []
    covmats.append(np.cov(list1))
    covmats.append(np.cov(list2))
    covmats.append(np.cov(list3))
    covmats.append(np.cov(list4))
    covmats.append(np.cov(list5))
            
    #print(covariance_sum)
            
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    covmat_inv = np.linalg.inv(covmat) 
    final_answer = np.zeros((len(ytest),5))
    for i in range(5):
        list_mean = [means[0][i],means[1][i]]
        for j in range(len(ytest)):
            list_test = [Xtest[j][0], Xtest[j][1]]
            intermediate_sub = np.subtract(list_test,list_mean); 
            final_answer[j][i] = np.dot(np.dot(np.transpose(intermediate_sub),covmat_inv),intermediate_sub)
    
    
    ypred = np.argmin(final_answer, 1) + 1
    acc = 0
    for i in range(len(ytest)):
        if ypred[i] == ytest[i]:
            acc = acc + 1
    acc = acc / len(ytest);     

    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    final_answer = np.zeros((len(ytest),5))
    for i in range(5):
        covmat_inv = np.linalg.inv(covmats[i]) 
        list_mean = [means[0][i],means[1][i]]
        for j in range(len(ytest)):
            list_test = [Xtest[j][0], Xtest[j][1]]
            intermediate_sub = np.subtract(list_test,list_mean); 
            final_answer[j][i] = np.dot(np.dot(np.transpose(intermediate_sub),covmat_inv),intermediate_sub)
            final_answer[j][i] = np.exp((-0.5) * final_answer[j][i])
            final_answer[j][i] = final_answer[j][i] / sqrt((np.linalg.det(covmats[i])))
            
    ypred = np.argmax(final_answer, 1) + 1
    acc = 0
    for i in range(len(ytest)):
        if ypred[i] == ytest[i]:
            acc = acc + 1        
    acc = acc / 100; 
        
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
	
    # IMPLEMENT THIS METHOD
    w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X),X)),np.transpose(X)),y)
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1         
    w = np.dot(np.dot(np.linalg.inv(np.add(lambd*np.identity(len(X[0])),np.dot(np.transpose(X),X))),np.transpose(X)),y)
    # IMPLEMENT THIS METHOD                                                   
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD
    without_transpose = np.subtract(ytest,np.dot(Xtest,w))
    mse = (np.dot(np.transpose(without_transpose),without_transpose))/ len(ytest)
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda    
    # IMPLEMENT THIS METHOD 
    w = np.reshape(w,(65,1))
    without_transpose = y - np.dot(X,w)
    error = (np.dot(np.transpose(without_transpose),without_transpose))/ 2
    
    error_grad = np.dot(np.transpose(np.dot(X,w) - y),X)
    error_grad = np.transpose(error_grad + np.transpose(lambd*w))
    return error,np.array(error_grad).flatten()

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (p+1)) 
    # IMPLEMENT THIS METHOD
    Xd = np.zeros((len(x),p+1))
    count = 0
    for j in x :
        for i in range(p+1):
            Xd[count][i] = pow(j,i)
        count = count + 1
    
    return Xd

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

print(np.shape(xx))
zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle_train = testOLERegression(w,X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_train_i = testOLERegression(w_i, X_i, y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))
print('MSE without intercept train '+str(mle_train))
print('MSE with intercept train '+str(mle_train_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1

print(np.min(mses3))
opt_lambda = lambdas[np.argmin(mses3)] 
print(opt_lambda)
print(mses3_train[6])

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4

k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 150}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
    
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()
print(np.min(mses4))
print(np.min(mses4_train))
print(lambdas[np.argmin(mses4)])
print(mses4[7])
print(mses4_train[7])

# Problem 5
pmax = 7
lambda_opt = opt_lambda # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
    
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
