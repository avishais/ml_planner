/*
 * plan_C_space.h
 *
 *  Created on: Nov 10, 2016
 *      Author: avishai
 */

#ifndef PLAN_C_SPACE_H_
#define PLAN_C_SPACE_H_

// OMPL
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/SE3StateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/config.h>

// Modified and custom planners
#include "CBiRRT_gd.h"
#include "RRT_gd.h"
#include "SBL_gd.h"

//#include "../validity_checkers/StateValidityCheckerGD.h"
#include "../validity_checkers/verification_class.h"

// Standard libraries
#include <iostream>
#include <fstream>

namespace ob = ompl::base;
namespace og = ompl::geometric;
using namespace std;
using namespace gd;

// An enum of available planners
enum plannerType
{
	PLANNER_BIRRT,
	PLANNER_RRT,
	PLANNER_SBL
};

bool isStateValid(const ob::State *state);

// Prototypes
class plan_C
{
public:

	bool plan(State c_start, State c_goal, double runtime, plannerType = PLANNER_BIRRT, double = 2);

	// Construct the planner specified by our command line argument.
	// This helper function is simply a switch statement.
	ob::PlannerPtr allocatePlanner(ob::SpaceInformationPtr, int, plannerType);

	bool solved_bool;
	double total_runtime;
	double maxStep;
};

#endif /* PLAN_C_SPACE_H_ */
