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

using namespace std;

int main(int argc, const char * argv[]) {
    
    
    //testing the Matrix class
    vector<vector<double>> tmp1;
    
    tmp1.resize(5);
    for ( int i{0}; i<tmp1.size(); i++) {
        tmp1[i].resize(10);
    }
    
    for ( int i{0}; i < 5; i++) {
        for(int j{0}; j< 10; j++) {
            tmp1[i][j] = 356;
        }
    }
    
    Matrix tmp2(tmp1);
    
    cout<< tmp2;
    
}
