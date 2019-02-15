# DenseLayersNeuralNetwork

Requires keras and Python3

#need four numpy matrices
#x_train = training features
#y_train = training class labels
#x_test = testing features
#y_test = testing class labels


#load training file (CSV file), file column should be class label (0 or 1)
#replace sys.argv[1] by file name
Train = np.loadtxt(sys.argv[1], delimiter=',')
print(Train.shape)
x_train = Train[:, 1:(Train.shape[1])]
y_train = Train[:, 0] 
print(x_train.shape)
print("class labels")
print(set(y_train))

#load testing file ((CSV file), file column should be class label (0 or 1)
#Replace sys.argv[2] by file name
Test = np.loadtxt(sys.argv[2], delimiter=',')  # this is our verification our model runs
print(Test.shape)
x_test = Test[:, 1:(Test.shape[1])]
y_test = Test[:, 0]
print(x_test.shape)
print(y_test.shape)



#perform 10-fold cross validation using 2 hidden layers and 100 nodes in each hidden layer
Metrics=layerDNN_kfold(x_train,y_train,x_test,y_test,2,100)
print("average testing metrics")
print(np.average(Metrics,axis=0))





