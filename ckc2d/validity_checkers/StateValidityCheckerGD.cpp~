/*
 * Checker.cpp
 *
 *  Created on: Oct 28, 2016
 *      Author: avishai
 */

/*
myStateValidityCheckerClass::myStateValidityCheckerClass(const ob::SpaceInformationPtr &si) {

}*/

#include "StateValidityCheckerGD.h"
#include <queue>

void gd::StateValidityChecker::defaultSettings()
{
	stateSpace_ = mysi_->getStateSpace().get();
	if (!stateSpace_)
		OMPL_ERROR("No state space for motion validator");
}

void gd::StateValidityChecker::retrieveStateVector(const ob::State *state, State &q) {
	// cast the abstract state type to the type we expect
	const ob::RealVectorStateSpace::StateType *Q = state->as<ob::RealVectorStateSpace::StateType>();

	for (unsigned i = 0; i < n; i++) {
		q[i] = Q->values[i]; // Set state of robot1
	}
}

void gd::StateValidityChecker::updateStateVector(const ob::State *state, State q) {
	// cast the abstract state type to the type we expect
	const ob::RealVectorStateSpace::StateType *Q = state->as<ob::RealVectorStateSpace::StateType>();

	for (unsigned i = 0; i < n; i++) {
		Q->values[i] = q[i];
	}
}

void gd::StateValidityChecker::printStateVector(const ob::State *state) {
	// cast the abstract state type to the type we expect
	const ob::RealVectorStateSpace::StateType *Q = state->as<ob::RealVectorStateSpace::StateType>();

	State q(n);

	for (unsigned i = 0; i < n; i++) {
		q[i] = Q->values[i]; // Set state of robot1
	}
	cout << "q: "; printVector(q);
}

bool gd::StateValidityChecker::IKproject(const ob::State *state, bool includeObs) {

	State q(n);
	retrieveStateVector(state, q);

	if (!IKproject(q, includeObs))
		return false;

	updateStateVector(state, q);
	return true;
}

bool gd::StateValidityChecker::IKproject(State &q, bool includeObs) {

	if (!GD(q))
		return false;

	q = get_GD_result();

	// Check Constraints
	if (include_constraints && !check_angles(q))
		return false;

	// Check Constraints
	if (includeObs && include_constraints && ( !self_collision(q) || !obstacle_collision(q) ))
		return false;

	return true;
}

State gd::StateValidityChecker::sample_q() {
	// c is a 12 dimensional vector composed of [q1 q2]

	State q(n);

	while (1) {
		// Randomly generate a chain
		for (int i = 0; i < n-1; i++)
			q[i] = fRand(-PI_, PI_);
		q[n-1] = fRand(0, 2*PI_);

		bool valid = GD(q);
		if (valid)
			break;
	}

	q = get_GD_result();

	// Check Constraints
	if (include_constraints && ( !check_angles(q) || !self_collision(q) || !obstacle_collision(q) )) {
		q[0] = -1000;
		return q;
	}

	return q;
}

// ------------------- Validity check

// Validates a state by switching between the two possible active chains and computing the specific IK solution (input) and checking collision
bool gd::StateValidityChecker::isValid(const ob::State *state) {

	isValid_counter++;

	State q(n);
	retrieveStateVector(state, q);

	if (!GD(q))
		return false;

	q = get_GD_result();

	if (MaxAngleDistance(q, q_prev) > RBS_tol)
		return false;

	// Check Constraints
	if (include_constraints && ( !check_angles(q) || !self_collision(q) || !obstacle_collision(q) ))
		return false;

	updateStateVector(state, q);
	q_prev = q;
	return true;
}

bool gd::StateValidityChecker::checkMotion(const ob::State *s1, const ob::State *s2)
{
	State q(n);
	// We assume motion starts and ends in a valid configuration - due to projection
	bool result = true;
	//int nd = stateSpace_->validSegmentCount(s1, s2);
	//cout << nd << endl;
	State q1(n), q2(n);
	retrieveStateVector(s1, q1);
	retrieveStateVector(s2, q2);
	int nd = normDistance(q1,q2)/dq;


	if (nd > 2)
	{
		retrieveStateVector(s1, q);
		q_prev = q;

		// temporary storage for the checked state
		ob::State *test = mysi_->allocState();

		for (int i = 1; i < nd; i++) {
			stateSpace_->interpolate(s1, s2, (double)i / (double)(nd-1), test);

			if (!isValid(test))
			{
				result = false;
				break;
			}
			//printStateVector(test);
		}

		retrieveStateVector(s2, q);
		if (MaxAngleDistance(q, q_prev) > 4*0.174533) //4*10deg
			result = false;

		mysi_->freeState(test);
	}

	return result;
}
double gd::StateValidityChecker::normDistance(State a1, State a2) {
	double sum = 0;
	for (int i=0; i < a1.size(); i++)
		sum += pow(a1[i]-a2[i], 2);
	return sqrt(sum);
}

double gd::StateValidityChecker::normVector(State q) {

	double sum;
	for (int i = 0; i < n; i++)
		sum += q[i]*q[i];

	return sqrt(sum);
}

double gd::StateValidityChecker::MaxAngleDistance(State a1, State a2) {
	double Max = 0;
	for (int i=0; i < a1.size()-1; i++)
		if (fabs(a1[i]-a2[i]) > Max)
			Max = fabs(a1[i]-a2[i]);
	return Max;
}

// ------------------------------- Constraints functions ---------------------------

bool gd::StateValidityChecker::check_angles(State q, double factor) {

	for (int i = 0; i < n-1; i++)
		if (q[i] > factor*get_qminmax() || q[i] < -factor*get_qminmax())
			return false;
	if (q[n-1] < 0)
		return false;

	return true;
}

bool gd::StateValidityChecker::self_collision(State q, double factor) {
	double Ax, Ay, Bx, By, Cx, Cy, Dx, Dy;
	State L = getL();
	Ax = Ay = 0;

	double cum_q = 0, cum_q_CD;
	for (int i = 0; i < n-3; i++) {
		cum_q += q[i];

		Bx = Ax + L[i]*cos(cum_q);
		By = Ay + L[i]*sin(cum_q);

		cum_q_CD = cum_q + q[i+1];
		Cx = Bx + L[i]*cos(cum_q_CD);
		Cy = By + L[i]*sin(cum_q_CD);
		for (int j = i+2; j < n-1; j++) {
			cum_q_CD += q[j];
			Dx = Cx + L[j]*cos(cum_q_CD);
			Dy = Cy + L[j]*sin(cum_q_CD);

			// ---- Check if two lines intersect ---
			double s1_x = Bx - Ax;
			double s1_y = By - Ay;
			double s2_x = Dx - Cx;
			double s2_y = Dy - Cy;
			double s = (-s1_y * (Ax - Cx) + s1_x * (Ay - Cy)) / (-s2_x * s1_y + s1_x * s2_y);
			double t = ( s2_x * (Ay - Cy) - s2_y * (Ax - Cx)) / (-s2_x * s1_y + s1_x * s2_y);

			double minLim = -0.2 * factor, maxLim = 1.2 * factor;
			if (s >= minLim && s <= maxLim && t >= minLim && t <= maxLim)
				return false;
			// -------------------------------------

			Cx = Dx;
			Cy = Dy;
		}

		Ax = Bx;
		Ay = By;
	}

	return true;
}

// Returns false if the lines AB and CD intersect, otherwise true.
// Currently only checks when lines are not parallel
bool gd::StateValidityChecker::LinesIntersect(State A, State B, State C, State D) {
	double s1_x, s1_y, s2_x, s2_y;
	s1_x = B[0] - A[0];
	s1_y = B[1] - A[1];
	s2_x = D[0] - C[0];
	s2_y = D[1] - C[1];

	double s, t;
	s = (-s1_y * (A[0] - C[0]) + s1_x * (A[1] - C[1])) / (-s2_x * s1_y + s1_x * s2_y);
	t = ( s2_x * (A[1] - C[1]) - s2_y * (A[0] - C[0])) / (-s2_x * s1_y + s1_x * s2_y);

	double minLim = -0.2, maxLim = 1.2;
	if (s >= minLim && s <= maxLim && t >= minLim && t <= maxLim)
		return false;

	return true; // No collision
}

bool gd::StateValidityChecker::obstacle_collision(State q, double factor) {
	double x, y;
	State L = getL();
	x = y = 0;

	double cum_q = 0;
	for (int i = 0; i < n-1; i++) {
		cum_q += q[i];

		x = x + L[i]*cos(cum_q);
		y = y + L[i]*sin(cum_q);

		for (int j = 0; j < obs.size(); j++) {
			if ( (x-obs[j][0])*(x-obs[j][0]) + (y-obs[j][1])*(y-obs[j][1]) < (obs[j][2]+factor*L[i])*(obs[j][2]+factor*L[i]) )
				return false;
		}
	}
	return true;
}

// ------------------------------------ RBS -------------------------------------------

// Validates a state by switching between the two possible active chains and computing the specific IK solution (input) and checking collision
bool gd::StateValidityChecker::isValidRBS(State& q) {

	isValid_counter++;

	if (!GD(q))
		return false;

	q = get_GD_result();

	// Check Constraints
	if (include_constraints && ( !check_angles(q) || !self_collision(q) || !obstacle_collision(q) ))
		return false;

	return true;
}

// Calls the Recursive Bi-Section algorithm (Hauser)
bool gd::StateValidityChecker::checkMotionRBS(const ob::State *s1, const ob::State *s2, int inter_inx)
{
	// We assume motion starts and ends in a valid configuration - due to projection
	bool result = true;

	State q1(n), q2(n);
	retrieveStateVector(s1, q1);
	retrieveStateVector(s2, q2);

	result = checkMotionRBS(q1, q2, 0, 0);

	return result;
}

// Implements local-connection using Recursive Bi-Section Technique (Hauser)
bool gd::StateValidityChecker::checkMotionRBS(State q1, State q2, int recursion_depth, int non_decrease_count, int inter_inx) {

	// Check if reached the required resolution
	double d = normDistance(q1,q2); // for joint limit distance heuristic
	//double d = normVector(angle_distance(q1, q2)); // for shortest distance heuristic
	if (d < RBS_tol)
		return true;

	if (recursion_depth > RBS_max_depth || non_decrease_count > 100)
		return false;

	State q_mid = midpoint(q1, q2);

	// Check obstacles collisions and joint limits
	if (!isValidRBS(q_mid)) // Also updates q_mid with the projected value
		return false;

	//if ( normVector(angle_distance(q1, q_mid)) > d || normVector(angle_distance(q_mid, q2)) > d) // Shortest angle distance
	if ( normDistance(q1, q_mid) > d || normDistance(q_mid, q2) > d) // joint limits
		non_decrease_count++;

	if ( checkMotionRBS(q1, q_mid, recursion_depth+1, non_decrease_count, 0) && checkMotionRBS(q_mid, q2, recursion_depth+1, non_decrease_count, 0) )
		return true;
	else
		return false;
}

State gd::StateValidityChecker::midpoint(State q1, State q2, int inter_inx) {

	State q_mid(n);

	//State v(n, 0);// = dec2bin(inter_inx, n-3);

	for (int i = 0; i < n; i++)
		q_mid[i] = (q1[i]+q2[i])/2;
		//q_mid[i] = midangle(q1[i], q2[i], v[i]); // If include_constraints=true, then this cannot be used and the simple (q1[i]+q2[i])/2 should be used.

	return q_mid;
}

// *************** Reconstruct the RBS - for post-processing and validation

// Reconstruct local connection with the Recursive Bi-Section algorithm (Hauser)
bool gd::StateValidityChecker::reconstructRBS(const ob::State *s1, const ob::State *s2, Matrix &Confs)
{
	State q1(n), q2(n);
	retrieveStateVector(s1, q1);
	retrieveStateVector(s2, q2);

	Confs.push_back(q1);
	Confs.push_back(q2);

	return reconstructRBS(q1, q2, Confs, 0, 1, 1, 0);
}

bool gd::StateValidityChecker::reconstructRBS(State q1, State q2, Matrix &M, int iteration, int last_index, int firstORsecond, int non_decrease_count) {
	// firstORsecond - tells if the iteration is from the first or second call for the recursion (in the last iteration).
	// last_index - the last index that was added to M.

	iteration++;

	// Check if reached the required resolution
	double d = normDistance(q1, q2);
	if (d < RBS_tol)
		return true;

	if (iteration > RBS_max_depth || non_decrease_count > 100)
		return false;

	State q_mid = midpoint(q1, q2);

	// Check obstacles collisions and joint limits
	if (!isValidRBS(q_mid)) // Also updates s_mid with the projected value
		return false;

	if ( normVector(angle_distance(q1, q_mid)) > d || normVector(angle_distance(q_mid, q2)) > d)
		non_decrease_count++;

	if (firstORsecond==1)
		M.insert(M.begin()+last_index, q_mid); // Inefficient operation, but this is only for post-processing and validation
	else
		M.insert(M.begin()+(++last_index), q_mid); // Inefficient operation, but this is only for post-processing and validation

	int prev_size = M.size();
	if (!reconstructRBS(q1, q_mid, M, iteration, last_index, 1, non_decrease_count))
		return false;
	last_index += M.size()-prev_size;
	if (!reconstructRBS(q_mid, q2, M, iteration, last_index, 2, non_decrease_count))
		return false;

	return true;
}

// ----------------------------------------------------------------------------

double gd::StateValidityChecker::midangle(double q1, double q2, int shortORlong) {
	// Returns the shortest (shortORlong=0)/ longest (shortORlong=1) angle distance.

	double q_mid;

	q1 < 0 ? q1 += 2*PI_ : q1+=0;
	q2 < 0 ? q2 += 2*PI_ : q2+=0;

	double dq = fabs(q1-q2);

	int sigma;
	q2 > q1 ? sigma = 1 : sigma = -1;

	if (!shortORlong) {
		if (dq > PI_) {
			dq = 2*PI_ - dq;
			sigma = -sigma;
		}
	}
	else {
		if (dq < PI_) {
			dq = 2*PI_ - dq;
			sigma = -sigma;
		}
	}

	return q1 + sigma * dq/2;
}

State gd::StateValidityChecker::angle_distance(State q1, State q2) {

	State dq(n);

	double q;
	for (int i = 0; i < n; i++) {
		q = q1[i] - q2[i] + PI_;
		while (q > 2*PI_)
			q -= 2*PI_;
		while (q < 0)
			q += 2*PI_;
		dq[i] = q - PI_;
	}

	/*for (int i = 0; i < n; i++) {
			dq[i] = fabs(q1[i]-q2[i]);
			if (dq[i] > PI_)
				dq[i] = 2*PI_ - dq[i];
		}*/
	//dq[i] = fmod((q1[i]-q2[i]) + PI_, 2*PI_) - PI_;

	return dq;

}

State gd::StateValidityChecker::dec2bin(int num, int k) {
	// k is the required size of the returned vector - pad with zeros on the left side.

	State b;

	do {

		if ( (num&1) == 0 )
			b.insert(b.begin(), 0);
		else
			b.insert(b.begin(), 1);

		num >>= 1;
	} while (num);

	while (b.size() < k)
		b.insert(b.begin(), 0);

	return b;
}



