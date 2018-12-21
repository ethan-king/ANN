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
string outputFilename = "Weights.txt";
double learningRate{0.005};
//contains normailzation values / max values in each column
vector<double> maxIOVal{};
//variables to hold training data
vector<vector<double>> input, output;
size_t hiddenNeuron{8};
//Training set partition size = 1 - validation size
double ptTrain{.80}, errTol{0.001};
unsigned int epochs{15};

//function prototypes
void loadTrainingCSV(const string&, vector<vector<double>> &, vector<vector<double>> &);
void forwardProp( const vector<double>&);
void backProp(const vector<double> &);




double random(double x){
    return (double)(rand() % 10000 + 1)/10000-0.5;
}

//double sigmoid(double x){
//    return 1/(1+exp(-x));
//}
//
//double sigmoidePrime(double x){
//    return exp(-x)/(pow(1+exp(-x), 2));
//}
//
//double stepFunction(double x){
//    if(x>0.9){
//        return 1.0;
//    }
//    if(x<0.1){
//        return 0.0;
//    }
//    return x;
//}

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
    
    //display menu
    //1- load csv, specify # of attributes
    //2- define hidden nodes
    //3- set learning rate
    //4- set activation function reLu or sigmoid
    //5- set partition  %
    //6 -
    
    unsigned int selection = 0, initialized=0; //menu selection
    
    cout << "Black Friday Neural Network Project (Codename Titan)\n" ;
    while (selection != 10) { // Creates a repeating menu of options unless 9 is chosen to ens the program

        cout<<"Current settings: "<<filename<< "; Hidden Nodes "<< hiddenNeuron <<"; Learning Rate "<< learningRate<<"; Epochs "<<epochs<<"; Partition Size "<<ptTrain<<"; Error "<< errTol <<endl;
        cout<<"-----------------------------------------"<<endl;
        
        cout << "Please enter one of the following:" << endl
        << "1) Run training" << endl
        << "2) Change file name for training" << endl
        << "3) Change number of nodes in hidden layer" << endl
        << "4) Change learning rate" << endl
        << "5) Change training epochs" << endl
        << "6) Change partition size for training set / test set" << endl
        << "7) Change error tolerance" << endl
        << "8) Run test" << endl
        << "9) Save weights"<<endl
        << "10) End program"<<endl
        << ">> ";
        
        cin >> selection;
        
        if (!cin){
            cout << "Incorrect input. Please try again.\n";
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
        }
        
        if (selection == 1) {
            cout << "\n\nInitializing...\n\n";
            loadTrainingCSV(filename, input, output);
            initialized = 1; //initilization marker that a training file has been used
            //LoadTraining(trainingFile); //Need to pass all loading of data to a function and use main only for menu and initialization
            
            
            //Turn our input and output matrices into Matrix class
            size_t inputNeuron = input[0].size();
            size_t outputNeuron = output[0].size();
            
            
            W1 = Matrix(inputNeuron, hiddenNeuron);
            W2 = Matrix(hiddenNeuron, outputNeuron);
            B1 = Matrix(1, hiddenNeuron);
            B2 = Matrix(1, outputNeuron);
            
            W1 = W1.applyFunction(random);
            W2 = W2.applyFunction(random);
            B1 = B1.applyFunction(random);
            B2 = B2.applyFunction(random);
            
            //iterate through given # of epochs
            cout<< setw(6)<< "Epoch " << setw(15)<< "Error"<<endl;
            
            double MSE{1};
            
            for (size_t h{0}; h<epochs && MSE > errTol; h++){
                //train NN on our input and output matrices
                double SSE{0};
                
                for( size_t i{0}; i< roundSzT( input.size()*ptTrain ); i++) {
                    //X is set to each vector in input
                    
                    forwardProp(input[i]);
                    //create dot prod vector to sum all entries
                    
                    //compare vs expected Output Y2
                    backProp(output[i]);
                    
                    SSE += 0.5*(Y2.sumElem() - Y.sumElem())*(Y2.sumElem() - Y.sumElem());
                    //double squaredError{(Y2-Y)*(Y2-Y)};
                }
                
                MSE = SSE/roundSzT( input.size()*ptTrain ); //
                
                cout<< setw(6)<< h <<setw(15)<< MSE <<endl;;
            }
            cout<<"\nNeural network trained!"<<endl;
        }
        else if (selection == 2) {
            
            cout << "\nPlease enter the new filename followed by its extension: ";
            string temp{""};
            while (!(cin>>temp)) {
                cout<<"Please enter a file name"<<endl;
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
            }
            ifstream infile(temp);
            if(!infile ) {
                cout<<"Error opening file"<<endl;
            }
            else {
                filename = temp;
                cout << "\nFilename changed to "<< filename<< "!"<<endl;
            }
            infile.close();
        }
        else if (selection == 3) {
            cout << "\nPlease enter the number of nodes for the neural network: ";
            
            //catching for error inputs
            while (!(cin >> hiddenNeuron)) {
                cout << "Incorrect input. Please try again.\n";
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
            }
            
            cout << "\nHidden neuron amount changed to "<< hiddenNeuron<< "!"<<endl;
            
        }
        else if (selection == 4) {
            cout << "\nPlease enter the new learning rate as a percent (ex: 0.05): ";
            
            while (!(cin >> learningRate) || (learningRate >1) || (learningRate <= 0)) {
                cout << "Input: (0,1]. Please try again.\n";
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
            }
            
            cout << "\nlearning rate changed to "<< learningRate<<"!\n";
        }
        else if (selection == 5) {
            cout << "\nPlease enter the number of epochs to train: ";
            
            while (!(cin >> epochs)) {
                cout << "Incorrect input. Please try again.\n";
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
            }
            
            if ( epochs > 9000) cout << "epochs are over 9000!"<< endl;
            cout << "\nepochs changed to "<< epochs<<"!\n";
            
        }
        
        else if (selection == 6) {
            cout << "\nPlease enter the partition size for the training set: ";
            
            while (!(cin >> ptTrain)) {
                cout << "Incorrect input. Please try again.\n";
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
            }
            
            cout << "\npartition size changed to "<< ptTrain<<"!\n";
            
        }
        
        else if (selection == 7) {
            cout << "\nPlease enter the value for error tolerance: ";
            
            while (!(cin >> errTol)) {
                cout << "Incorrect input. Please try again.\n";
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
            }
            
            cout << "\nError tolerance changed to "<< errTol<<"!\n";
            
        }
        else if (selection == 8) {
            if (initialized == 0) {
                cout<<"Must train the NN before validation. Please run option 1 first."<<endl;
            }
            else {
                cout <<"\nRunning training set through the neural network"<<endl;
                //Run validation set through updated NN
                double SSE {0};
                int ctr{0};
                cout<< setw(11)<< "Validation" << setw(15)<< "Error"<<endl;
                
                for( size_t i{roundSzT(input.size()*ptTrain)}; i < input.size(); i++){
                    
                    forwardProp(input[i]);
                    
                    Y2 = Matrix({output[i]}); //expected output
                    
                    //create dot prod vector to sum all entries
                    cout<< setw(11) << i << setw(15) << (Y2.sumElem() - Y.sumElem())*(Y2.sumElem() - Y.sumElem())<<endl;
                    SSE += 0.5*(Y2.sumElem() - Y.sumElem())*(Y2.sumElem() - Y.sumElem());
                    
                    ctr++;
                }
                
                double MSE = SSE/ctr;
                cout<< setw(11)<<"MSE"<<setw(15) << MSE <<endl;
                
                cout<<"done."<<endl;
            }
        
            
        }
        
        
        else if (selection == 9) {
            cout << "\nPlease enter a file name for the weights file: ";
            
            while (!(cin >> outputFilename)) {
                cout << "Incorrect filename. Please try again.\n";
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
            }
            
            ofstream weightOut( outputFilename, ios::out);
            if (!weightOut){
                cerr<< "File could not be opened"<< endl;
                exit(EXIT_FAILURE);
            }
            
            weightOut << "W1"<<endl;
            W1.write(weightOut);
            
            weightOut <<endl<< "B1"<<endl;
            B1.write(weightOut);
            
            weightOut <<endl<< "W2"<<endl;
            W2.write(weightOut);
            
            weightOut <<endl<< "B2"<<endl;
            B2.write(weightOut);
            
            
            
            cout << "\nWeights saved to "<<outputFilename<<"!\n";
            
        }
        
        else if (selection == 10)
            cout << "\nGoodbye";
//        else if (selection > 8) {
//            cout<< "Please enter a valid menu selection"<<endl;
//            cin>> selection;
//        }
        
        
        
//        else {
//            if (!(cin >> selection)) {
//                cout << "Incorrect input. Please try again.\n";
//                cin.clear();
//                cin.ignore(numeric_limits<streamsize>::max(), '\n');
//            }
//        }
        
        else cout << "Please enter a valid menu selection"<<endl<<endl;
    }
}

void loadTrainingCSV(const string &filename, vector<vector<double>> &input, vector<vector<double>> &output){
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
 
        vector<vector<double>> maxValVect{maxIOVal};
        
        fileStream.close();
        cout<< "Successfully loaded "<<filename<<" into the input and output vectors"<<endl;
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

}

void forwardProp( const vector<double>& in) {
    X = Matrix({in});
    H = X.dot(W1).add(B1).applyFunction(reLu);
    Y = H.dot(W2).add(B2).applyFunction(reLu);
}

void backProp( const vector<double>& out) {
    Y2 = Matrix({out});
    
    dJdB2 = (Y-Y2)*((H.dot(W2)+(B2)).applyFunction(reLuPrime));
    dJdB1 = dJdB2.dot(W2.transpose())*((X.dot(W1)+B1).applyFunction(reLuPrime));
    dJdW2 = H.transpose().dot(dJdB2);
    dJdW1 = X.transpose().dot(dJdB1);
    
    // update weights
    W1 = W1-(dJdW1*learningRate);
    W2 = W2-(dJdW2*learningRate);
    B1 = B1-(dJdB1*learningRate);
    B2 = B2-(dJdB2*learningRate);
}
