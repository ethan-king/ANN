//
//  main.cpp
//  ANN
//
//  Created by Ethan on 11/28/18.
//  Copyright © 2018 Ethan. All rights reserved.
//

#include <iostream>
#include "Matrix.hpp"
#include <vector>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cmath>


using namespace std;

Matrix X, W1, H, W2, Y, B1, B2, Y2, dJdB1, dJdB2, dJdW1, dJdW2;
string filename = "BFData.csv";
double learningRate{0.005};



//function prototypes
void forwardProp( const vector<double>&);


void loadTrainingCSV(const char* filename, vector<vector<double>> &input, vector<vector<double>> &output){
}

double random(double x){
    return (double)(rand() % 10000 + 1)/10000-0.5;
}

double sigmoid(double x){
    return 1/(1+exp(-x));
}

double sigmoidePrime(double x){
    return exp(-x)/(pow(1+exp(-x), 2));
}

double stepFunction(double x){
    if(x>0.9){
        return 1.0;
    }
    if(x<0.1){
        return 0.0;
    }
    return x;
}

double reLu(double x){
    if (x < 0) return 0;
    else return x;
}

double reLuPrime(double x){
    if (x >0) return 1;
    else return 0;
}

size_t roundSzT(double x) {
    if(x - static_cast<size_t>(x) <.5) return static_cast<size_t>(x);
    else return static_cast<size_t>(x)+1;
}


int main(int argc, const char * argv[]) {
    
    //contains normailzation values / max values in each column
    vector<double> maxIOVal{};
    
    
    //variables to hold training data
    vector<vector<double>> input, output;
    
    //open training data csv file
    ifstream fileStream(filename);
    //########### MUST SPECIFY number of attributes in .csv file
    int numAttr{8};
    maxIOVal.resize(numAttr);
    
    //test if file is opened
    if (!fileStream) {
        cerr << "File "<< filename<< " could not be opened" << endl;
        exit(EXIT_FAILURE);
    }
    
    //write csv to input and output matrices, update maxValVect
    if(fileStream) {
        cout<<filename<<" opened."<<endl;
        string temp{""}, a{""}; //strings to hold lines from .csv, then individual values in the line
        while( getline(fileStream, temp) ) { //read each line of the .csv
            int attrCtr{0}; //token counter - when Ctr reaches the end, place value into outputv
            //cout<<temp<<endl;
            vector<double> tempVect{}, tempVect2{}; //holds vectors for input and output
            //parse the line by commas
            for( stringstream sst(temp); getline(sst, a, ',');) {
                //check vs vector of maximum values and update if the current number is greater
                if(stod(a) > maxIOVal[attrCtr]) {
                    maxIOVal[attrCtr] = stod(a);
                }
                // if the is the final value in the line, add it to the output vector
                if (attrCtr == numAttr-1 ) {
                    try{
                        tempVect2.push_back(stod(a));
                    }
                    catch(invalid_argument e) {
                        cout<< e.what()<<endl;
                    }
                    
                }
                else{ //else add the value to the input vector
                    try{
                        tempVect.push_back(stod(a));
                    }
                    catch(invalid_argument e) {
                        cout<< e.what()<<endl;
                    }
                    attrCtr++;
                }
            }
            input.push_back(tempVect);
            output.push_back(tempVect2);
        }
        
        cout<< "Input vetor:"<<endl;
        cout<< input<< endl;
        cout<< "Output vector:"<<endl;
        cout<< output<< endl;
        
        vector<vector<double>> maxValVect{maxIOVal};
        cout<< "Max Values:" << endl;
        cout<< maxValVect<<endl;
        
        fileStream.close();
    }
    
    
    //Normalize all terms by dividing by Max Value
    
    //input & output
    for( int i{0}; i<maxIOVal.size(); i++ ){ //iterate over maxVal vector
        for( int j{0}; j<input.size(); j++) { //iterate over input vector<vector<double>>
            input[j][i] /= maxIOVal[i];
        }
    }
    
    // divide all values in output by last value in maxVal vector
    
    for( int i{0}; i< output.size(); i++) {
        output[i][0] /= *(maxIOVal.end()-1);
    }
    
    cout<<"Input after normalization:"<<endl;
    cout<< input<<endl;
    cout<<"Output after normalization:"<<endl;
    cout<< output<< endl;
    
    
    //Turn our input and output matrices into Matrix class
    
    int inputNeuron = input[0].size();
    int outputNeuron = output[0].size();
    int hiddenNeuron{8};
    


    W1 = Matrix(inputNeuron, hiddenNeuron);
    W2 = Matrix(hiddenNeuron, outputNeuron);
    B1 = Matrix(1, hiddenNeuron);
    B2 = Matrix(1, outputNeuron);
    
    W1 = W1.applyFunction(random);
    W2 = W2.applyFunction(random);
    B1 = B1.applyFunction(random);
    B2 = B2.applyFunction(random);
    
    
    //compute Output Y
    //X is first vector in input
    
    //Training set partition size = 1 - validation size
    double partition{.80};
    
    
    //iterate through given # of epochs
    cout<< setw(6)<< "Epoch " << setw(15)<< "Error"<<endl;
    int epochs{20};
    
    for (size_t h{0}; h<epochs; h++){
        //train NN on our input and output matrices
        double SSE{0};
        
        for( size_t i{0}; i< roundSzT( input.size()*partition ); i++) {
            //X is set to each vector in input
            
            forwardProp(input[i]);
            
            Y2 = Matrix({output[i]}); //expected output
            
            //create dot prod vector to sum all entries
            
            
            SSE += (Y2.sumElem() - Y.sumElem())*(Y2.sumElem() - Y.sumElem());
            
            
            //compare vs expected Output Y2
            
            dJdB2 = (Y-Y2)*((H.dot(W2)+(B2)).applyFunction(reLuPrime));
            dJdB1 = dJdB2.dot(W2.transpose())*((X.dot(W1)+B1).applyFunction(reLuPrime));
            dJdW2 = H.transpose().dot(dJdB2);
            dJdW1 = X.transpose().dot(dJdB1);
            
            // update weights
            W1 = W1-(dJdW1*learningRate);
            W2 = W2-(dJdW2*learningRate);
            B1 = B1-(dJdB1*(learningRate));
            B2 = B2-(dJdB2*(learningRate));
            
            //double squaredError{(Y2-Y)*(Y2-Y)};
        }
        
        double MSE = SSE/roundSzT( input.size()*partition ); //
        
        cout<< setw(6)<< h <<setw(15)<< MSE <<endl;;
    }
    
    //Run validation set through updated NN
    double SSE {0};
    int ctr{0};
    cout<< setw(11)<< "Validation" << setw(15)<< "Error"<<endl;

    for( size_t i{roundSzT(input.size()*partition)}; i < input.size(); i++){
        
        
        forwardProp(input[i]);
        
        Y2 = Matrix({output[i]}); //expected output
        
        //create dot prod vector to sum all entries

        SSE += (Y2.sumElem() - Y.sumElem())*(Y2.sumElem() - Y.sumElem());
        ctr++;
    }
    
    double MSE = SSE/ctr;
    
    cout<< setw(26) << MSE <<endl;
    
    
    cout<<"done."<<endl;
    return 0;
    
    
    
}


void forwardProp( const vector<double>& in) {
    
    X = Matrix({in});
    
    H = X.dot(W1).add(B1).applyFunction(reLu);
    Y = H.dot(W2).add(B2).applyFunction(reLu);

}

