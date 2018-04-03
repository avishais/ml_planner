/*
 * Checker.h
 *
 *  Created on: Oct 28, 2016
 *      Author: avishai
 */

#ifndef CHECKER_H_
#define CHECKER_H_

#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/SE3StateSpace.h>
#include <ompl/base/StateValidityChecker.h>
#include "ompl/base/MotionValidator.h"
#include "ompl/base/State.h"
#include <ompl/config.h>

#include "kdl_class.h"
#include "apc_class.h"

#include <iostream>


namespace ob = ompl::base;
using namespace std;

class StateValidityChecker : public kdl, public ckc
{
public:
	StateValidityChecker(const ob::SpaceInformationPtr &si, int joint_num, int passive_chains_num, double custom_num) : mysi_(si.get()), kdl(joint_num, custom_num), ckc(joint_num, custom_num) {

		n = joint_num;
		m = passive_chains_num;
		q_prev.resize(n);
		L = ckc::getL();

	};
	StateValidityChecker(int joint_num, int passive_chains_num, double custom_num=-1) : kdl(joint_num, custom_num), ckc(joint_num, custom_num) { //Constructor
		n = joint_num;
		m = passive_chains_num;
		L = ckc::getL();
	};

	/** Project a configuration in the ambient space to the constraint surface */
	bool IKproject(const ob::State *, bool = true);
	bool IKproject(State &, bool = true);
	bool IKprojectAPC(State &, int, int);

	/** Identify the IK solutions of a configuration using all passive chains defined */
	State identify_state_ik(const ob::State *);
	State identify_state_ik(State);

	/** Validity check using standard OMPL */
	bool isValid(const ob::State *);

	/** Validity check for a vector<double> type  */
	bool isValidRBS(State&, int, int);

	/** Serial local connection check  */
	bool checkMotion(const ob::State *, const ob::State *);

	/** Recursive Bi-Section local connection check  */
	bool checkMotionRBS(const ob::State *, const ob::State *, int, int);
	bool checkMotionRBS(State, State, int active_chain, int ik_sol, int recusion_depth, int);

	/** Reconstruct a local connection using RBS for post-processing  */
	bool reconstructRBS(const ob::State *, const ob::State *, Matrix &, int, int);
	bool reconstructRBS(State, State, int, int, Matrix &, int, int, int, int);

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

	double get_RBS_tol() {
		return RBS_tol;
	}

	// Performance parameters
	int isValid_counter;
	int get_isValid_counter() {
		return isValid_counter;
	}

	int n, m;

	// Include constraints?
	const bool include_constraints = true; // Enable/Disable constraints
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

	double L;
	double dq = 0.1;
	double RBS_tol = 0.05; // RBS local connection resolution
	int RBS_max_depth = 100; // Maximum RBS recursion depth

	ompl::RNG rng_;
};





#endif /* CHECKER_H_ */
