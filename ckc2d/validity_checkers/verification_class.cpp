/*
 * veification_class.cpp
 *
 *  Created on: Aug 1, 2017
 *      Author: avishai
 */

#include "verification_class.h"

bool verification_class::verify_path() {

	cout << "Verifying path in file: " << path_file << "..." << endl;

	State q(n);
	Matrix M;
	int tmp;

	ifstream fq;
	fq.open(path_file);

	fq >> tmp;

	int i = 0;
	while(!fq.eof()) {
		for (int j=0; j < n; j++) {
			fq >> q[j];
		}
		M.push_back(q);
		i++;
	}
	fq.close();

	bool vrf = verify_path(M);

	if (vrf) {
		cout << "***************************************" << endl;
		cout << "************ PATH FEASIBLE ************" << endl;
		cout << "***************************************" << endl;
	}
	else {
		cout << "***************************************" << endl;
		cout << "*********** ! PATH FAILED ! ***********" << endl;
		cout << "***************************************" << endl;
	}

	return vrf;
}

bool verification_class::verify_path(Matrix M) {

	int m = M.size();

	// Validate joint limits
	for (int i = 0; cAngles && i < m; i++) {
		if (!check_angles(M[i], 1.15)) {
			cout << "***************************************" << endl;
			cout << "@ Joint limits failure!" << endl;
			cout << "@ Node " << i << endl;
			return false; // Report joint limit breach
		}
	}

	// Validate continuity
	for (int i = 1; i < m; i++) {
		// cout << normDistance(M[i],M[i-1]) << endl;
		if (normDistance(M[i],M[i-1]) > continuity_tol) {
			cout << "***************************************" << endl;
			cout << "@ Discontinuity failure!" << endl;
			cout << "@ Node " << i << endl;
			return false; // Report discontinuity
		}
	}

	// Validate closure constraint
	for (int i = 0; i < m; i++) {
		if (!test_constraint(M[i])) {
			cout << "***************************************" << endl;
			cout << "@ Closure constraint failure!" << endl;
			cout << "@ Node " << i << endl;
			return false;
		}
	}

	// Validate obstacle collisions
	for (int i = 0; cObs && i < m; i++) {
		if (!obstacle_collision(M[i], 0.2)) {
			cout << "***************************************" << endl;
			cout << "@ Obstacle collision failure!" << endl;
			cout << "@ Node " << i << endl;
			return false;
		}
	}

	// Validate sel collisions
	for (int i = 0; cColl && i < m; i++) {
		if (!self_collision(M[i], 0.7)) {
			cout << "***************************************" << endl;
			cout << "@ Self collision failure!" << endl;
			cout << "@ Node " << i << endl;
			return false;
		}
	}

	return true; // No errors
}

void verification_class::log_path_file(Matrix M) {

	std::ofstream mf;

	mf.open(path_file);

	mf << M.size() << endl;

	for (int i = 0; i < M.size(); i++) {
		for (int j = 0; j < M[0].size(); j++)
			mf << M[i][j] << " ";
		mf << endl;
	}

	mf.close();

	/*State L = getL();
	mf.open("../paths/path_info.txt");
	mf << n << endl << 0 << endl << L[0] << endl << get_bx() << endl << get_by() << endl << get_qminmax() << endl;
	mf.close();*/
}

bool verification_class::test_constraint(State q) {

	State C = kdl::constraint(q);

	bool verify = true;
	for (int j = 0; j < C.size() && verify; j++)
		if (fabs(C[j]) > 0.05)
			verify = false;

	//if (!verify)
	//	printVector(C);

	return verify;

}




