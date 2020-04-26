# PROGRAMMER: Alfonso Manzano Gonzalez
# COURSE: CSCI 4336.01 
# DATE:   Feb 19, 2020
# ASSIGNMENT:  Homework 1. the preceptron
# ENVIRONMENT: Windows 10
# FILES INCLUDED: main
# PURPOSE: Implement the PLA algorithm(perceptron)
# INPUT:  Dataset for iris flower. Using only colum 0 and 2 as inputs.
# PRECONDITIONS: Predict the specie of the flower which is given in column 4
# OUTPUT: claculate the weights of the perceptron and predict the type of flower
# POSTCONDITIONS: none
# ALGORITHM:  
# 1. load dataset
# 2. Extract column 0, 2 as inputs and 4 as ouput
# 3. Split the data in 80% for training and 20% for testing

# 5. perceptron
    # make a vector for the weight with values 0. [0,0,0]
    # for every record in the training set
    #     make a vector such as [1, x1, x2] where 1 is the bias, and x1, x2 are the inputs in colum0 and colum2
    #     dot product of the weights vector and current x's values
    #     check if element is miscalssified or not. classified if dot product is=>0  
    #     if the element is miscalssified claculate new weights:
    # return the weights

# ERRORS:  none
# EXAMPLE: input: 5.9,7.0 output: 1 (Iris-versicolor)

    


class Perceptron:
    #------------constructor----------#
    def __init__(self):
        self.weights=[0,0,0]#initial weights
        self.counter=0
     
    #------------function to train the preceptron------------------#    
    def train_perceptron(self, data_set, target):
        # since the data set is linearly separable iterate until we find the correct weights
        #for the whole data_set
        while (self.counter!=len(data_set)):
            self.counter=0
            for i in range (len(data_set)):
                #get the inputs x1 and x2 ->[x1,x2]
                #insert 1 as bias -> [1, x1, x2]
                x_n=data_set[i].copy()
                x_n.insert(0,1)
                
                #multiply w*xn (dot product)
                dot_product=sum(a*b for a,b in zip(x_n,self.weights))
                
                #check if the element is misclassified
                if (dot_product>=0):
                    label=1
                else:
                    label=-1
                    
                #if element is classified correctly count++
                #otherwise claculate new weights and break
                if label==target[i]:
                    self.counter=self.counter+1
                else:
                    #new weights
                    for j in range (len(self.weights)):
                        self.weights[j]=self.weights[j]+(target[i]*x_n[j]) 
                    break
        #return weights
        return self.weights
    
    #---------------for testing the model and get accurancy--------#
    def test_perceptron(self, data_set, target):
        counter=0
        #for all the element in the lists
        for i in range (len(data_set)):
            #predict the result using the the weights
            prediction=self.prediction(data_set[i][0],data_set[i][1])
            #if the prediction is correct counter++
            if prediction==target[i]:counter+=1
        #return the percentage of the correct predictions
        return counter/len(data_set)*100
    
    #---------------for using the model--------#
    def prediction(self, x1, x2):
        #activation function
        val=self.weights[0]+(x1*self.weights[1])+(x2*self.weights[2])
        if (val>=0):return 1
        else: return -1


if __name__=='__main__':
    training_set=[[5.1,3.5,1.4,0.2,'Iris-setosa'],
            [4.6,3.4,1.4,0.3,'Iris-setosa'],
            [5.0,3.4,1.5,0.2,'Iris-setosa'],
            [4.4,2.9,1.4,0.2,'Iris-setosa'],
            [4.9,3.1,1.5,0.1,'Iris-setosa'],
            [5.4,3.7,1.5,0.2,'Iris-setosa'],
            [4.8,3.4,1.6,0.2,'Iris-setosa'],
            [4.8,3.0,1.4,0.1,'Iris-setosa'],
            [4.3,3.0,1.1,0.1,'Iris-setosa'],
            [5.8,4.0,1.2,0.2,'Iris-setosa'],
            [5.7,4.4,1.5,0.4,'Iris-setosa'],
            [5.4,3.9,1.3,0.4,'Iris-setosa'],
            [5.1,3.5,1.4,0.3,'Iris-setosa'],
            [5.7,3.8,1.7,0.3,'Iris-setosa'],
            [5.1,3.8,1.5,0.3,'Iris-setosa'],
            [5.4,3.4,1.7,0.2,'Iris-setosa'],
            [5.1,3.7,1.5,0.4,'Iris-setosa'],
            [4.6,3.6,1.0,0.2,'Iris-setosa'],
            [5.1,3.3,1.7,0.5,'Iris-setosa'],
            [4.8,3.4,1.9,0.2,'Iris-setosa'],
            [5.0,3.0,1.6,0.2,'Iris-setosa'],
            [5.0,3.4,1.6,0.4,'Iris-setosa'],
            [5.2,4.1,1.5,0.1,'Iris-setosa'],
            [5.5,4.2,1.4,0.2,'Iris-setosa'],
            [4.9,3.1,1.5,0.1,'Iris-setosa'],
            [5.0,3.2,1.2,0.2,'Iris-setosa'],
            [5.5,3.5,1.3,0.2,'Iris-setosa'],
            [4.9,3.1,1.5,0.1,'Iris-setosa'],
            [4.4,3.0,1.3,0.2,'Iris-setosa'],
            [5.1,3.4,1.5,0.2,'Iris-setosa'],
            [5.0,3.5,1.3,0.3,'Iris-setosa'],
            [4.5,2.3,1.3,0.3,'Iris-setosa'],
            [4.4,3.2,1.3,0.2,'Iris-setosa'],
            [5.0,3.5,1.6,0.6,'Iris-setosa'],
            [5.1,3.8,1.9,0.4,'Iris-setosa'],
            [4.8,3.0,1.4,0.3,'Iris-setosa'],
            [5.1,3.8,1.6,0.2,'Iris-setosa'],
            [4.6,3.2,1.4,0.2,'Iris-setosa'],
            [5.3,3.7,1.5,0.2,'Iris-setosa'],
            [5.0,3.3,1.4,0.2,'Iris-setosa'],
            [7.0,3.2,4.7,1.4,'Iris-versicolor'],
            [6.4,3.2,4.5,1.5,'Iris-versicolor'],
            [6.9,3.1,4.9,1.5,'Iris-versicolor'],
            [5.5,2.3,4.0,1.3,'Iris-versicolor'],
            [6.5,2.8,4.6,1.5,'Iris-versicolor'],
            [5.7,2.8,4.5,1.3,'Iris-versicolor'],
            [6.3,3.3,4.7,1.6,'Iris-versicolor'],
            [4.9,2.4,3.3,1.0,'Iris-versicolor'],
            [6.6,2.9,4.6,1.3,'Iris-versicolor'],
            [5.6,2.9,3.6,1.3,'Iris-versicolor'],
            [6.7,3.1,4.4,1.4,'Iris-versicolor'],
            [6.1,2.8,4.0,1.3,'Iris-versicolor'],
            [6.3,2.5,4.9,1.5,'Iris-versicolor'],
            [6.1,2.8,4.7,1.2,'Iris-versicolor'],
            [6.4,2.9,4.3,1.3,'Iris-versicolor'],
            [6.6,3.0,4.4,1.4,'Iris-versicolor'],
            [6.8,2.8,4.8,1.4,'Iris-versicolor'],
            [6.7,3.0,5.0,1.7,'Iris-versicolor'],
            [6.0,2.9,4.5,1.5,'Iris-versicolor'],
            [5.7,2.6,3.5,1.0,'Iris-versicolor'],
            [5.5,2.4,3.8,1.1,'Iris-versicolor'],
            [5.5,2.4,3.7,1.0,'Iris-versicolor'],
            [5.8,2.7,3.9,1.2,'Iris-versicolor'],
            [6.0,2.7,5.1,1.6,'Iris-versicolor'],
            [5.4,3.0,4.5,1.5,'Iris-versicolor'],
            [6.0,3.4,4.5,1.6,'Iris-versicolor'],
            [6.7,3.1,4.7,1.5,'Iris-versicolor'],
            [6.3,2.3,4.4,1.3,'Iris-versicolor'],
            [5.6,3.0,4.1,1.3,'Iris-versicolor'],
            [5.5,2.5,4.0,1.3,'Iris-versicolor'],
            [5.5,2.6,4.4,1.2,'Iris-versicolor'],
            [6.1,3.0,4.6,1.4,'Iris-versicolor'],
            [5.8,2.6,4.0,1.2,'Iris-versicolor'],
            [5.0,2.3,3.3,1.0,'Iris-versicolor'],
            [5.6,2.7,4.2,1.3,'Iris-versicolor'],
            [5.7,3.0,4.2,1.2,'Iris-versicolor'],
            [5.7,2.9,4.2,1.3,'Iris-versicolor'],
            [6.2,2.9,4.3,1.3,'Iris-versicolor'],
            [5.1,2.5,3.0,1.1,'Iris-versicolor'],
            [5.7,2.8,4.1,1.3,'Iris-versicolor']]
    
    testing_set=[[4.9,3.0,1.4,0.2,'Iris-setosa'],
                [4.7,3.2,1.3,0.2,'Iris-setosa'],
                [4.6,3.1,1.5,0.2,'Iris-setosa'],
                [5.0,3.6,1.4,0.2,'Iris-setosa'],
                [5.4,3.9,1.7,0.4,'Iris-setosa'],
                [5.2,3.5,1.5,0.2,'Iris-setosa'],
                [5.2,3.4,1.4,0.2,'Iris-setosa'],
                [4.7,3.2,1.6,0.2,'Iris-setosa'],
                [4.8,3.1,1.6,0.2,'Iris-setosa'],
                [5.4,3.4,1.5,0.4,'Iris-setosa'],
                [5.2,2.7,3.9,1.4,'Iris-versicolor'],
                [5.0,2.0,3.5,1.0,'Iris-versicolor'],
                [5.9,3.0,4.2,1.5,'Iris-versicolor'],
                [6.0,2.2,4.0,1.0,'Iris-versicolor'],
                [6.1,2.9,4.7,1.4,'Iris-versicolor'],
                [5.6,3.0,4.5,1.5,'Iris-versicolor'],
                [5.8,2.7,4.1,1.0,'Iris-versicolor'],
                [6.2,2.2,4.5,1.5,'Iris-versicolor'],
                [5.6,2.5,3.9,1.1,'Iris-versicolor'],
                [5.9,3.2,4.8,1.8,'Iris-versicolor'],]
            
    
    
    
    
    #extract column 0, 2 as features and 4 as target Iris-setosa=-1, Iris-versicolor=1 from training and testing
    #training
    training_features=[]
    training_target=[]
    for row in training_set :
        training_features.append([row[0], row[2]])
        if row[4]=='Iris-setosa':
            training_target.append(-1)
        else:training_target.append(1)

    #testing
    testing_features=[]
    testing_target=[]
    for row in testing_set :
        testing_features.append([row[0], row[2]])
        if row[4]=='Iris-setosa':
            testing_target.append(-1)
        else:testing_target.append(1)
    
    #create the perceptron
    perceptron=Perceptron()
    
    #train the perceptron
    weights=perceptron.train_perceptron(training_features,training_target)
    
    #test and calculate the accuracy
    accuracy=perceptron.test_perceptron(testing_features,testing_target)
    
    #print the results
    print("The weights are: "+str(weights))
    print("The accuracy is: "+str(accuracy)+"%")
    
    #make a prediction
    guess=perceptron.prediction(5.9,7.0 )
    species='Iris-setosa' if guess==-1 else 'Iris-versicolor'
    print("prediction: "+species)
    
    
    
    
    




