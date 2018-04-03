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
#include <ompl/util/RandomNumbers.h>

#include "apc_class.h"

#include <iostream>

namespace ob = ompl::base;
using namespace std;

class StateValidityChecker : public ckc
{
public:
	StateValidityChecker(const ob::SpaceInformationPtr &si, int joints_num, int passive_chains_num, double custom_num) : mysi_(si.get()), ckc(joints_num, custom_num) { //Constructor
		n = joints_num;
		m = passive_chains_num;
		q_temp.resize(6);
		L = getL();
	};
	StateValidityChecker(int joints_num, double custom_num=-1) : ckc(joints_num, custom_num) { //Constructor
		n = joints_num;
		m = n - 2;
		q_temp.resize(6);
		L = getL();
	};

	// Validity check
	bool isValid(const ob::State *state, int active_chain, int IK_sol, bool project);
	bool checkMotion(const ob::State *s1, const ob::State *s2, int active_chain, int ik_sol);
	bool checkMotionRBS(const ob::State *s1, const ob::State *s2, int active_chain, int ik_sol);
	bool isValidRBS(Vector&, int active_chain, int IK_sol);
	bool RBS(Vector, Vector, int active_chain, int ik_sol, int recusion_depth, int);
	Vector midpoint(Vector q1, Vector q2);
	Vector angle_distance(Vector q1, Vector q2);
	//bool checkMotionRBS(const ob::State *s1, const ob::State *s2, int active_chain, int ik_sol);
	bool RBS(const ob::State *, const ob::State *, int active_chain, int ik_sol, int recusion_depth, ob::State *);

	// Retrieve and update
	void retrieveStateVector(const ob::State *state, Vector &q);
	void retrieveStateVector(const ob::State *state, Vector &q, Vector &ik);
	void updateStateVector(const ob::State *state, Vector q);
	void updateStateVectorIK(const ob::State *state, Vector ik);
	void updateStateVector(const ob::State *state, Vector q, Vector ik);
	void printStateVector(const ob::State *state);

	void defaultSettings();
	double normDistance(Vector, Vector);
	double normVector(Vector q);
	Vector identify_state_ik(const ob::State *state);
	bool IKproject(ob::State *state, int, int);
	bool InterMotion(const ob::State *s1, const ob::State *s2, Matrix &interPath, int active_chain, int ik_sol);
	void interpolate(const ob::State *s1, const ob::State *s2, double s, const ob::State *test);
	
	/** Reconstruct a local connection using RBS for post-processing  */
	bool reconstructRBS(const ob::State *, const ob::State *, Matrix &, int, int);
	bool reconstructRBS(Vector, Vector, int, int, Matrix &, int, int, int, int);

	Vector sample_q();

	int get_valid_solution_index() {
		return valid_solution_index;
	}

	// Performance parameters
	int isValid_counter;
	int get_isValid_counter() {
		return isValid_counter;
	}

	double iden = 0;
	double get_iden() {
		return iden;
	}
	
	double get_RBS_tol() {
		return RBS_tol;
	}

	// Include constraints?
	const bool include_constraints = true; // Enable/Disable constraints
	bool check_angles(Vector, double = 1);
	bool self_collision(Vector, double = 1);
	bool obstacle_collision(Vector, double = 0.3);
	bool LinesIntersect(Vector A, Vector B, Vector C, Vector D, double);
	//Matrix obs = {{-4, 4, 1},{3.4, 7.5, 1}, {4.2, -2.9, 1.5}, {8, 2, 0.8}};
	Matrix obs = {{-3.2, 4, 1},{2.6, 6.8, 1}, {4.2, -2.9, 1.5}, {8, 2, 0.8}};

	double t3, t4, t5, t6, t7, t8;
	
private:
	double L;
	ob::StateSpace *stateSpace_;
	ob::SpaceInformation *mysi_;
	Vector q_temp;
	int valid_solution_index;
	int n;
	int m;
	const double dq = 0.1;
	
	double RBS_tol = 0.1; // RBS local connection resolution
	int RBS_max_depth = 60; // Maximum RBS recursion depth

	ompl::RNG rng_;
};





#endif /* CHECKER_H_ */
