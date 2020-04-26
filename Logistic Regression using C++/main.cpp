/*
PROGRAMMER: Alfonso Manzano Gonzalez
COURSE: CSCI 4336
DATE:   Apr 27, 2020
ASSIGNMENT:  Final Project Diabetes Diagnosis using Logistic regression
ENVIRONMENT: Windows 10
FILES INCLUDED: Zip with file with project using visual studio and dataset
ENVIRONMENT: Windows 10
FILES INCLUDED: Zip with project
PURPOSE:  The purpose of this project is to develop from scratch Logistic Regression
          for diagnosing diabetes.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
//for shuffle the vector
#include <random>
#include <algorithm>
#include <chrono>


#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
//for shuffle the vector
#include <random>
#include <algorithm>
#include <chrono>


using namespace std;

//--------------------------function used for operations.----------------//
void Write(){
    ofstream myFileOut ("test.txt");
    if (myFileOut.is_open()){
        myFileOut<<"this is a test"<<endl;
        myFileOut<<"Alfonso"<<endl;
        myFileOut.close();
    }else{cout<<"Unable to open the file";}
}
//prints the contents of a multidimensional vector
void printVector(vector<vector<double>> &myVector){
    cout<<"[ ";
    for(int i=0; i<myVector.size(); i++){
        for (int j=0; j<myVector[i].size();j++){
            cout<<setprecision(4)<<myVector[i][j]<<"  ";
        }
        cout<<" ]"<<endl;
    }
}
//prints content of a vector
void printVector(vector<double> &myVector){
    cout<<"[ ";
    for(int i=0; i<myVector.size(); i++){
        cout<<setprecision(4)<<myVector[i]<<" ";
    }
    cout<<" ]"<<endl;
}
//prints the first elements of a multidimensional vector
void printVectorHead(vector<vector<double>> &myVector, int head=5){
    cout<<"\nHead: \n";
    if (head>myVector.size()){head=myVector.size();}
    for(int i=0; i<head; i++){
        for (int j=0; j<myVector[i].size();j++){
            cout<<setprecision(15)<<myVector[i][j]<<" ";
        }
        cout<<endl;
    }
}
//prints the first elements of a multidimensional vector
void printVectorHead(vector<double> &myVector, int head=5){
    cout<<"\nHead: \n";
    if (head>myVector.size()){head=myVector.size();}
    for(int i=0; i<head; i++){
        cout<<setprecision(15)<<myVector[i]<<" ";
    }
    cout<<endl;
}

//read file and return vector with features
vector< vector<double>> readCSV(string file="diabetes.csv", int NumFeatures=5){
    vector<vector<double>> features={};//vector for storing all the features and return them
    // try to open the file
    ifstream myFileIn (file);
    // if the file is open
    //read the features
    if(myFileIn.is_open()){
        cout<<"--->Reading dataset from file...\n";
        //while we can read from the file
        string line;
        //get the line and turn it in a stream of chars
        while (getline(myFileIn, line, '\n')){
            istringstream tempLine(line);
            string value;
            vector<double> temp;
            //extract the values from the stream and push on a temp vector
            while(getline(tempLine, value,',')){
                temp.push_back(atof(value.c_str()));
            }
            //push every record vertor
            features.push_back(temp);
        }

        //close the file
        myFileIn.close();
        cout<<"--->File processed...\n";
        //if the file was no open print error
    }else{
        cout<<"Unable to open the file";
    }
    //return the vector of vectors double
    return features;
}
//return the max value for the specific column in a multidimensional vector
double maxNum(vector<vector<double>> &arr, int col){
    double max=arr[0][col];
    int size=arr.size();
    for (int i=0; i<size;i++){
        if (arr[i][col]>max){
            max=arr[i][col];
        }
    }
    return max;
}
//return the min value for the specific column in a multidimensional vector
double minNum(vector<vector<double>> &arr, int col){
    double min=arr[0][col];
    int size=arr.size();
    for (int i=0; i<size;i++){
        if (arr[i][col]<min){
            min=arr[i][col];
        }
    }
    return min;
}
//return the number of values zeros for the specific column in a multidimensional vector
int numZeros(vector<vector<double>> &arr, int col) {
    int total=0;
    int size=arr.size();
    for (int i=0; i<size;i++){
        if (arr[i][col]==0){
            total++;
        }
    }
    return total;
}
//return the number of values not zeros for the specific column in a multidimensional vector
int numNotZeros(vector<vector<double>> &arr, int col) {
    int total=0;
    int size=arr.size();
    for (int i=0; i<size;i++){
        if (arr[i][col]!=0){
            total++;
        }
    }
    return total;
}
//return the mean for the specific column in a multidimensional vector
int colMean(vector<vector<double>> &arr, int col){
    double sum=0.0;
    double count=0;
    int size=arr.size();
    for (int i=0; i<size;i++){
        sum=sum+arr[i][col];
        count++;
    }
    double mean=sum/count;
    return mean;
}
//return a vector with max, mean, zeros, notzeros for a given multidimensional vector
void statistics(vector<vector<double>> &data, vector<double> &max, vector<double> &min, vector<double> &mean, vector<double> &zeros,vector<double> &notZeros){
    int size = data[0].size();
    int numFeatures= data[0].size();
    for (int i=0;i<size;i++){
        max.push_back(maxNum(data,i));
        min.push_back(minNum(data,i));
        zeros.push_back(numZeros(data,i));
        notZeros.push_back(numNotZeros(data,i));
        mean.push_back(colMean(data,i));
    }
}
//scales the entry values for diagnois
vector<double> scaleData(vector<vector <double>> data, vector<double> &entry){
    vector <double> x;
    vector<double> max, min, mean, zeros, notZeros;
    statistics(data,max,min,mean,zeros, notZeros);
    for (int col=0; col<entry.size(); col++){
        //scale de data using x-min/max-min
        x.push_back((entry[col]-min[col])/(max[col]-min[col]));
    }
    return x;
}
//split the data for training and testing
void splitDataSet(vector<vector<double>> &data, vector<vector<double>> &xTrain, vector<double>&yTrain,vector<vector<double>> &xTest, vector<double>&yTest , float sizeOfXtest=0.8){
    cout<<"--->Preparing Dataset...\n";
    int proportion=data.size()*sizeOfXtest;
    int seed=5432;  //1675 //1770
    //shuffle dataset
    shuffle(begin(data),end(data), default_random_engine(seed));
    //TrainSet
    cout<<"--->Cleaning Data...\n";
    vector<double> max, min, mean, zeros, notZeros;
    statistics(data,max,min,mean,zeros, notZeros);
    for (int row=0; row<data.size();row++){
        //insert bias 1 to the dataset
        vector<double> tempFeatures={1};
        //clean data
        for (int col=0; col<data[row].size()-1; col++){
            //insert mean to features with a lot of missing values
           if (data[row][col]==0){data[row][col]=mean[col];}
            //scale de data using x-min/max-min
            tempFeatures.push_back((data[row][col]-min[col])/(max[col]-min[col]));
        }
        double tempTarget={data[row][8]};
        if (row < proportion) {
            xTrain.push_back(tempFeatures);
            yTrain.push_back(tempTarget);
        } else {
            xTest.push_back(tempFeatures);
            yTest.push_back(tempTarget);
        }
    }
    cout<<"--->Dataset ready...\n";
}
//return the dot product of two vectors
vector<double> dot (vector<vector<double>> &a, vector<double> &b){
    vector<double> dot;
    for (int i=0; i<a.size();i++){
        double sum=0;
        for(int j=0;j<b.size(); j++){
            double prod=a[i][j]*b[j];
            sum=sum+prod;
        }
        dot.push_back(sum);
    }
    return dot;
}
//returns a vector with the difference of two vectors
vector<double> sub (vector<double> a, vector<double> b){
    vector<double> subtraction;
    for (int i=0; i<a.size();i++){//max 3
        subtraction.push_back(a[i]-b[i]);
    }
    return subtraction;
}
//performs division for two vectors
vector<double> divide (vector<double> a, double b){
    vector<double> div;
    for (int i=0; i<a.size();i++){
        div.push_back(a[i]/b);
    }
    return div;
}
//mutiply a vector for a given number
vector<double> multiply (vector<double> a, double b){
    vector<double> mul;
    for (int i=0; i<a.size();i++){
        mul.push_back(a[i]*b);
    }
    return mul;
}
//transpose a multidimensional vector
vector<vector<double>> transpose(vector<vector<double>> matrix){
    vector<vector<double>> trans;
    for (int i =0; i<matrix[0].size();i++){
        vector<double> temp;
        for (int j=0; j<matrix.size(); j++){
            temp.push_back(matrix[j][i]);
        }
        trans.push_back(temp);
    }
    return trans;
}

//------class for the logistic regression-----------/
class LogisticRegression{
public:
    double lR;
    int iter;
    double errorIn=0;
    vector<double> w;
//constructor
    LogisticRegression(double learningRate=6, int iterations=600  ){
        lR=learningRate;
        iter=iterations;
    }
//reset default values for learning rate and total of iterations
    void resetValues(){
        lR=6;
        iter=600;
    };
//performs the sigmoid function for a given vector
    vector <double> sigmoid(vector<double>&z){
        vector<double> h;
        for (int i=0; i<z.size();i++){
            double e=1/(1+exp(-z[i]));
            h.push_back(e);
        }
        return h;
    }
//predicts in 0 or 1 for a vector with features
    vector<double> predict(vector <vector <double>> &X){
        vector<double> prediction;
        prediction=dot(X,w);
        for (int i=0; i<prediction.size();i++){
            if(prediction[i]>=0.5){
                prediction[i]=1;
            }else{prediction[i]=0;}
        }
        return prediction;
    }
//predicts positive or negative for a manual entry by scaling
//and the predict the corresponding target
    string predictEntry(vector<double> &data, vector<double> max, vector<double> min){
        //initialize with bias
        vector<double> temp={1};
        for (int col=0; col<data.size(); col++){
            //scale  data using x-min/max-min
            temp.push_back((data[col]-min[col])/(max[col]-min[col]));
        }
        vector<vector<double>> tempFeatures;
        tempFeatures.push_back(temp);
        vector<double> pred=predict(tempFeatures);
        if (pred[0]==0){return "Negative";}
        else{return "Positive";}
    }
//return the accuracy for the model based on predicted vs. actual values
    double accuracy(vector<double> &yPredictions, vector<double> &Y ){
        double count=0;
        for(int i = 0; i<yPredictions.size();i++){
            if (yPredictions[i]==Y[i]){
                count++;
            }
        }
        return 100*(count/yPredictions.size());
    }
//train the model using logistic regression
    void fit(vector<vector<double>> &X, vector<double> &y ){
        //initialize w to zeros
        w.clear();
        for(int i=0;i<X[0].size();i++){
            w.push_back(0);
        }
        for (int i=0; i<iter;i++){
            //calculate z
            vector<double> z=dot(X,w);
            //calculate h
            vector<double> h=sigmoid(z);

            //calculate gradient
            vector<double> s=sub(h,y);//h(x)-y
            vector<vector<double>> t=transpose(X);//x.T
            vector<double> p=dot(t,s);//x.T*(h(x)-y)
            vector<double> gradient=divide(p,y.size());//gradient=x.T*(h(x)-y)/N
            //update weights w=w-(gradient*learningRate)
            vector<double> tempProd=multiply(gradient, lR );
            w=sub(w,tempProd);
        }
        //calculate errorIn
        vector<double> Yprediction=predict(X);
        errorIn=accuracy(Yprediction,y);
    }
};


//-----------------------functions for interface------------------//
void printHeader(){
    cout<<"\t++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
        <<"\t-------------   Diabetes Diagnosis   ---------------\n"
        <<"\t-------------           with         ---------------\n"
        <<"\t-------------   Logistic Regression  ---------------\n"
        <<"\t++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n";
}
//prints the main menu
void printMainMenu(){
    cout<<"\nWhat would you like to do?\n"
        <<"[1] Model Training Report\n"
        <<"[2] Dataset Report\n"
        <<"[3] Diagnose Using Model\n"
        <<"[4] Retrain Model\n"
        <<"[5] Reset Model\n"
        <<"[6] Quit\n";
}
//returns selections from the main menu
int mainMenu(){
    int selection=0;
    while(true) {
        printMainMenu();
        cin>>selection;
        if (cin.fail()){
            cout<<"wrong entry!"<<endl;
            cin.clear();
            cin.ignore(1000,'\n');
        }else{
            break;
        }
    }
    return selection;
}
//validate entry from user for options
double validEntry(string s){
    double entry;
    while(true) {
        cout<<s;
        cin >> entry;
        if (cin.fail()) {
            cout << "wrong entry!" << endl;
            cin.clear();
            cin.ignore(1000, '\n');
        } else {
            break;
        }
        cout<<endl;
    }
    return entry;
}



//prints column for report
void printReportColumn(vector<double> &myVector){

    for(int i=0; i<myVector.size(); i++){
        cout<<setw(13)<<setprecision(4)<<myVector[i];
    }
    cout<<endl;
}

int main() {

    //vectors for dataset
    //features and targets
    vector<vector<double>> xTrain, xTest;
    vector<double > yTrain, yTest;

    //print header
    printHeader();

    //read file with data
    vector<vector<double>> dataSet=readCSV();

    //statistics for dataset report
    vector<double> max, min, mean, zeros, notZeros;
    statistics(dataSet,max,min,mean,zeros, notZeros);

    //split data fro training and testing
    splitDataSet(dataSet,xTrain,yTrain,xTest,yTest,0.8 );

    //create object fro logistic regression
    auto start =chrono::steady_clock::now();//start time
    LogisticRegression LR(6,600);

    //fit the data
    cout<<"--->Training model...\n";
    LR.fit(xTrain,yTrain);
    auto end = chrono::steady_clock::now();//finish time

    //training time
    auto trainingTime=end-start;
    cout<<"--->Model Ready to Use!\n";


    //loop for options
    bool quit=false;
    while(quit==false){
        //get menu selection
        int selection=mainMenu();
        enum {ModelReport=1, DatasetReport,Diagnose, Retrain, ResetModel, Quit};

        //-------options------//
        //Current model report option
        if (selection==ModelReport){
            cout<<"Training time: "<< chrono::duration_cast<chrono::milliseconds>(end - start).count()<< " ms." << endl;
            cout<<"Accuracy on training data: "<<setprecision(4)<<LR.errorIn<<"%"<<endl;
            vector <double> Ypred=LR.predict(xTest);
            double acc=LR.accuracy(Ypred,yTest);
            cout<<"Accuracy on testing data: "<<setprecision(4)<<acc<<"%"<<endl;
            cout<<"Weights: ";
            printVector(LR.w);

            //dataset report option
        }else if(selection==DatasetReport){
            cout<<"Number of entries: "<<dataSet.size()<<endl;
            int n=13;
            cout<<"\n\t ";
            cout<<left<<setw(n)<<setw(n)<<"Pregnant"<<setw(n)<<"Glucose"<<setw(n)<<"BloodPressure"<<setw(n)<<"Skin"<<setw(n)
                <<"Insulin"<<setw(n)<<"BMI"<<setw(n)<<"Pedigree"<<setw(n)<<"Age"<<setw(n)<<"Target"<<endl;
            cout<<"Max:     ";
            printReportColumn(max);
            cout<<"Min:     ";
            printReportColumn(min);
            cout<<"Mean:    ";
            printReportColumn(mean);
            cout<<"Zeros:   ";
            printReportColumn(zeros);

            //diagnose option
        }else if(selection==Diagnose){
            double entry;
            vector<double>t;
            //{15	136	70	32	110	37.1	0.15	43	1};
            entry=validEntry("Please enter the number of pregnancies: ");
            t.push_back(entry);
            entry=validEntry("Please enter the glucose: ");
            t.push_back(entry);
            entry=validEntry("Please enter the blood pressure: ");
            t.push_back(entry);
            entry=validEntry("Please enter the skin thickness: ");
            t.push_back(entry);
            entry=validEntry("Please enter the insulin: ");
            t.push_back(entry);
            entry=validEntry("Please enter the BMI: ");
            t.push_back(entry);
            entry=validEntry("Please enter the number for the pedigree function: ");
            t.push_back(entry);
            entry=validEntry("Please enter the age: ");
            t.push_back(entry);
            string diag=LR.predictEntry(t,max,min);
            cout<<"\nDiagnosis---> "<<diag<<endl;

            //retrain model option
        }else if(selection==Retrain){
            LR.iter=validEntry("Please enter the number of Iterations: ");
            LR.lR=validEntry("Please enter the learning rate: ");
            LR.fit(xTrain,yTrain);
            cout<<"Model retrained, go to option [1] to see report"<<endl;

            //reset default values option
        }else if(selection==ResetModel){
            LR.resetValues();
            LR.fit(xTrain,yTrain);
            cout<<"Model rest, go to option [1] to see report"<<endl;

            //quit
        }else if(selection==Quit){
            cout<<"\nThank you. Good bye!"<<endl;
            quit= true;

            //wrong selection
        }else{cout<<"Wrong entry!!!"<<endl;}

    }
}
