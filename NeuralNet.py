import numpy as np
import scipy
import scipy.io
import pickle 

class Neural_Net():
    def __init__(self):
        self.W1 = None
        self.W2 = None
        
    def train(self, X, y, params):
        #params = [hidden_func, output_func, h_func_prime, o_func_prime, hidden layer size, number of labels, learning_rate, iterations]
        self.inputLayerSize = X.shape[1]  # 785
        self.hiddenLayerSize = params[4]  # 200
        self.outputLayerSize = params[5]  # 10
        self.learning_rate = params[6]
        iterations = params[7]
        
        if self.W1 is None:
            self.W1 = .001 * np.random.randn(self.inputLayerSize+1, self.hiddenLayerSize) # (785 x 200)
            self.W2 = .001 * np.random.randn(self.hiddenLayerSize + 1, self.outputLayerSize) # (201 x 10) 
            self.W1[-1,:] = 0
            self.W2[-1,:] = 0
        
        self.hidden_func = params[0]
        self.output_func = params[1]
        self.h_func_prime = params[2]
        self.o_func_prime = params[3]
        
        y_one = y
        y = self.convert_Labels(y)
        loss = []
        accuracy = []
        for k in range(iterations):
            i = np.random.randint(X.shape[0])
            dJdW1, dJdW2 = self.meanSquaredPrime(X[i], y[i])
            #dJdW1, dJdW2 = self.crossEntropyPrime(X[i], y[i])
            
            grad =  np.concatenate((dJdW1.ravel(), dJdW2.ravel()))
            self.W1 -= self.learning_rate * dJdW1 #(785x1)(1x200) = (785 x 200) 
            self.W2 -= self.learning_rate * dJdW2 #(201x1)(1x10) = (201 x 10)
        
    def meanSquaredPrime(self, x, y):
        #forward
        yHat = self.predict2(x)
        dJdy = yHat - y
        
        #backward
        delta3 = dJdy * self.o_func_prime(self.z3) # (1x10) * (1x10) = (1x10)
        w2 = np.delete(self.W2, 0, 0)
        delta2 = np.dot(delta3, w2.T) * self.h_func_prime(self.z2) #(1x10)(10x200) * (1x200) = (1x200) 
        
        #reshaping
        self.a1.shape = (1, self.inputLayerSize + 1) 
        self.a2.shape = (1, self.hiddenLayerSize + 1) 
        delta2.shape = (1, self.hiddenLayerSize)
        delta3.shape = (1, self.outputLayerSize) 
        
        dJdW1 = np.dot(self.a1.T, delta2)
        dJdW2 = np.dot(self.a2.T, delta3)
        
        return dJdW1, dJdW2
        
    def crossEntropyPrime(self, x, y):
        #forward
        yHat = self.predict2(x)

        #backward
        delta3 = yHat - y
        w2 = np.delete(self.W2, 0, 0)
        delta2 = np.dot(delta3, w2.T) * self.h_func_prime(self.z2)
        
        #reshaping
        self.a1.shape = (1, self.inputLayerSize + 1) 
        self.a2.shape = (1, self.hiddenLayerSize + 1) 
        delta2.shape = (1, self.hiddenLayerSize)
        delta3.shape = (1, self.outputLayerSize) 
        
        dJdW1 = np.dot(self.a1.T, delta2)
        dJdW2 = np.dot(self.a2.T, delta3)
        
        return dJdW1, dJdW2
        
    def predict(self, x):
        a1 = np.insert(x, 0, 1)
        z2 = np.dot(a1, self.W1)
        a2 = self.hidden_func(z2)
        a2 = np.insert(a2, 0, 1)
        z3 = np.dot(a2, self.W2)
        y = self.output_func(z3)
        return np.argmax(y)
        
    def predict2(self, x):
        self.a1 = np.insert(x, 0, 1) #activation at the first layer is just our example with an extra bias feature(1 x 785)
        self.z2 = np.dot(self.a1, self.W1) #activity of second layer given by the activation times the hidden layer weights (1 x 785)(785 x 200) = (1 x 200)
        self.a2 = self.hidden_func(self.z2) #activation at second layer is the activation function applied to the activity (1 x 200) 
        self.a2 = np.insert(self.a2, 0, 1) # prepend a "1" as a bias term to our activation (1 x 201) 
        self.z3 = np.dot(self.a2, self.W2) # output activity is (1 x 201)(201 x 10)  = (1 x 10)
        yHat = self.output_func(self.z3) # output vector is the output activation function applied to the output activity (1 x 10)
        return yHat
    
    def meanSquared(self, x, y): # square mean
        yHat = self.predict2(x)
        y = self.convert_Label(y)
        J = 0.5*sum((y-yHat)**2)
        return J
    
    def crossEntropy(self, x, y):
        yHat = self.predict2(x)
        y = self.convert_Label(y)
        J = 0
        for i in range(len(y)):
            J -= (y[i]*np.log(np.abs(yHat[i])) + (1 - y[i])*np.log(np.abs(1 - yHat[i])))
        return J
        
    def computeLoss(self, X, y):
        loss = 0
        for i in range(len(y)):
            loss += self.meanSquared(X[i], y[i])
            #loss += self.crossEntropy(X[i], y[i])
        return loss/float(len(y))
        
    def convert_Labels(self, y):
        labels = []
        label_vector = [0]*self.outputLayerSize
        for label in y:
            a = label_vector[:]
            a[label] = 1
            labels.append(a)
            
        return labels
    
    def convert_Label(self, y):
        label_vector = [0]*self.outputLayerSize
        label_vector[y] = 1
        return label_vector
        
    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize* (self.inputLayerSize +1)
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize+1, self.hiddenLayerSize))
        W2_end = W1_end + (self.hiddenLayerSize+1)*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize+1, self.outputLayerSize))
        
    def computeNumericalGradients(self, x, y):
        #y = self.convert_Labels(y)
        paramsInit = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        numgrad = np.zeros(paramsInit.shape)
        perturb = np.zeros(paramsInit.shape)
        e = 1e-4
        
        for p in range(len(paramsInit)):
            perturb[p] = e
            self.setParams(paramsInit + perturb)
            loss2 = self.meanSquared(x,y)
            
            self.setParams(paramsInit - perturb)
            loss1 = self.meanSquared(x,y)
            numgrad[p] = (loss2 - loss1)/(2*e)
            
            perturb[p] = 0
            
        self.setParams(paramsInit)
        
        return numgrad

#### Activation Functions ####
def sigmoid(z):
    return 1/(1 +np.exp(-z))
    
def tanh(z):
    return np.tanh(z)
    
def sigmoid_Prime(z):
    return np.exp(-z)/((1 + np.exp(-z))**2)
    
def tanh_Prime(z):
    return 1 - np.tanh(z)**2
     
    
     
#### Get data and shuffle it. ####
trainingData= scipy.io.loadmat("train.mat")
testData= scipy.io.loadmat("test.mat")
labels1 = np.ravel(trainingData["train_labels"])
images1 = np.reshape(trainingData["train_images"].transpose(), (60000, 784))
test_images = np.reshape(testData["test_images"].transpose(), (10000, 784))

labels = []
images = []
indexShuf = range(60000)
np.random.shuffle(indexShuf)
for i in indexShuf:
    labels.append(labels1[i])
    images.append(images1[i])
    
images -= np.mean(images, axis = 0)
images /= 255

net = Neural_Net()

if __name__ == '__main__':
    learning_rate = .04 #.06 is good for large iterations
    iterations = 300000
    net.train(images, labels, [sigmoid, tanh, sigmoid_Prime, tanh_Prime, 200, 10, learning_rate, iterations])
    fd = open('net.pkl', 'wb')
    pickle.dump(net, fd)
    fd.close()
    count = 0
    for i in range(60000):
        yHat = net.predict(images[i])
        if yHat == labels[i]:
            count += 1
        
    print count/float(60000), learning_rate, iterations, "Mean Squared Training Accuracy"
    
    for i in range(len(test_images)):
        print net.predict(test_images[i])








