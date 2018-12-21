//
//  Matrix.hpp
//  ANN
//
//  Created by Ethan on 11/28/18.
//  Copyright © 2018 Ethan. All rights reserved.
//

#ifndef Matrix_hpp
#define Matrix_hpp

#include <vector>
#include <iostream>
#include <fstream>

class Matrix
{
public:
    Matrix();
    Matrix(int height, int width);
    Matrix(std::vector<std::vector<double> > const &array);
    
    Matrix multiply(double const &value); // scalar multiplication
    Matrix operator*(double const &); 
    Matrix add(Matrix const &m) const; // addition
    Matrix operator+(Matrix const&); // + operator
    Matrix subtract(Matrix const &m) const; // subtraction
    Matrix operator-(Matrix const&);
    Matrix multiply(Matrix const &m) const; // hadamard product
    Matrix operator*(Matrix const &);
    Matrix dot(Matrix const &m) const; // dot product
    Matrix transpose() const; // transposed matrix
    
    Matrix applyFunction(double (*function)(double)) const; // to apply a function to every element of the matrix
    
    void print(std::ostream &flux) const; // pretty print of the matrix
    void write(std::ofstream&);
    
    double sumElem() const;
    
private:
    std::vector<std::vector<double> > array;
    size_t height;
    size_t width;
};

std::ostream& operator<<(std::ostream &flux, Matrix const &m); // overloading << operator to print easily

#endif /* Matrix_hpp */
