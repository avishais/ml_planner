/*
 * net_proj.h
 *
 *  Created on: Feb 27, 2017
 *      Author: avishai
 */

#ifndef NET_PROJ_H_
#define NET_PROJ_H_

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>

#include <eigen/Dense>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

typedef vector<double> State;
typedef vector<vector< double >> Matrix;

class net_proj
{
private:
	vector<MatrixXd> W;
	vector<VectorXd> b;

	int n; // Number of hidden layers
	int n_encoded; // Number of layers of the encoder
	int n_decoded; // Number of layers of the decoder

	float x_max, x_min; // For normalization

public:
	// Constructor
	net_proj();

	void import_weights();

	VectorXd activation(VectorXd);

	State encoder(State);
	State decoder(State);

	VectorXd normalize(VectorXd);
	VectorXd denormalize(VectorXd);

	// void printWeight(Matrix); 
	// void printBias(State); 
	void printWeight(MatrixXd); 
	void printBias(VectorXd); 

};



#endif /* NET_PROJ_H_ */
