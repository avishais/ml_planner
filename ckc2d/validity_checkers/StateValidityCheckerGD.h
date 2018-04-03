/*
 * Checker.h
 *
 *  Created on: Oct 28, 2016
 *      Author: avishai
 */

#ifndef CHECKER_H_GD
#define CHECKER_H_GD

#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/SE3StateSpace.h>
#include <ompl/base/StateValidityChecker.h>
#include "ompl/base/MotionValidator.h"
#include "ompl/base/State.h"
#include <ompl/config.h>

#include "kdl_class.h"

#include <iostream>

namespace ob = ompl::base;
using namespace std;

namespace gd {

class StateValidityChecker : public kdl
{
public:
	StateValidityChecker(const ob::SpaceInformationPtr &si, int joint_num, double custom_num) : mysi_(si.get()), kdl(joint_num, custom_num) {

		n = joint_num;
		q_prev.resize(n);

	};
	StateValidityChecker(int joint_num, double custom_num=-1) : kdl(joint_num, custom_num) { //Constructor
		n = joint_num;
	};

	/** Project a configuration in the ambient space to the constraint surface */
	bool IKproject(const ob::State *, bool = true);
	bool IKproject(State &, bool = true);

	/** Validity check using standard OMPL */
	bool isValid(const ob::State *);

	/** Validity check for a vector<double> type  */
	bool isValidRBS(State&);

	/** Serial local connection check  */
	bool checkMotion(const ob::State *, const ob::State *);

	/** Recursive Bi-Section local connection check  */
	bool checkMotionRBS(const ob::State *, const ob::State *, int = 0);
	bool checkMotionRBS(State, State, int, int, int = 0);

	/** Reconstruct a local connection using RBS for post-processing  */
	bool reconstructRBS(const ob::State *, const ob::State *, Matrix &);
	bool reconstructRBS(State, State, Matrix &, int, int, int, int);

	State midpoint(State, State, int = 0);
	double midangle(double, double, int);
	State angle_distance(State, State);
	State dec2bin(int, int);

	/** Retrieve state from ob::State to vector<double> */
	void retrieveStateVector(const ob::State *state, State &q);

	/** Update state to ob::State from vector<double> */
	void updateStateVector(const ob::State *state, State q);

	/** Print ob::State to console */
	void printStateVector(const ob::State *state);

	/** Set default OMPL setting */
	void defaultSettings();

	/** Calculate norm distance between two vectors */
	double normDistance(State, State);
	double normVector(State q);

	double MaxAngleDistance(State a1, State a2);

	/** Sample a feasible configuration */
	State sample_q();

	void log_q(State q, bool New);

	double get_RBS_tol() {
		return RBS_tol;
	}

	// Performance parameters
	int isValid_counter;
	int get_isValid_counter() {
		return isValid_counter;
	}

	int n;

	// Include constraints?
	const bool include_constraints = false; // Enable/Disable constraints
	bool check_angles(State, double = 1);
	bool self_collision(State, double = 1);
	bool obstacle_collision(State, double = 0.3);
	bool LinesIntersect(State A, State B, State C, State D);
	//Matrix obs = {{-4, 4, 1},{3.4, 7.5, 1}, {4.2, -2.9, 1.5}, {8, 2, 0.8}};
	Matrix obs = {{-3.2, 4, 1},{2.6, 6.8, 1}, {4.2, -2.9, 1.5}, {8, 2, 0.8}};

private:
	ob::StateSpace *stateSpace_;
	ob::SpaceInformation    *mysi_;
	State q_prev;


	double dq = 0.1;
	double RBS_tol = 0.05; // RBS local connection resolution
	int RBS_max_depth = 100; // Maximum RBS recursion depth

	ompl::RNG rng_;

};

}



#endif /* CHECKER_H_GD */
