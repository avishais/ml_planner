/*
 * veification_class.cpp
 *
 *  Created on: Aug 1, 2017
 *      Author: avishai
 */

#ifdef PCS
#include "StateValidityCheckerPCS.h"
#endif
#ifdef PGD
#include "StateValidityCheckerGD.h"
#endif
#ifdef HB
#include "StateValidityCheckerHB.h"
#endif

//#include <stdio.h>
#include <iostream>
#include <vector>
#include <math.h>

#ifdef PCS
using namespace pcs;
#endif
#ifdef PGD
using namespace gd;
#endif

typedef vector<double> State;
typedef vector<vector< double >> Matrix;

class verification_class : public StateValidityChecker
{
public:
	verification_class(int joints_num, double custom_num = 0.3) : StateValidityChecker(joints_num, joints_num, custom_num)
	{
		cout << "Initiated verification module." << endl;
		continuity_tol = get_RBS_tol() * 1.1;

		n = joints_num;
	}

	bool verify_path();
	bool verify_path(Matrix M);
	void log_path_file(Matrix M);
	bool test_constraint(State q);

private:
	double L;

	int n;

	double continuity_tol;
	
	string path_file = "./paths/path.txt";

	bool cAngles = true; // Check angle limits?
	bool cObs = true; // Check collisions with obstacles?
	bool cColl = true; // Check self collisions?
};


