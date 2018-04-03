/*
 * Checker.cpp
 *
 *  Created on: Oct 28, 2016
 *      Author: Avishai Sintov
 */

/*
myStateValidityCheckerClass::myStateValidityCheckerClass(const ob::SpaceInformationPtr &si) {

}*/

#include "StateValidityCheckerPCS.h"
#include <queue>

void pcs::StateValidityChecker::defaultSettings()
{
	stateSpace_ = mysi_->getStateSpace().get();
	if (!stateSpace_)
		OMPL_ERROR("No state space for motion validator");
}

void pcs::StateValidityChecker::retrieveStateVector(const ob::State *state, State &q) {
	// cast the abstract state type to the type we expect
	const ob::RealVectorStateSpace::StateType *Q = state->as<ob::RealVectorStateSpace::StateType>();

	for (unsigned i = 0; i < n; i++) {
		q[i] = Q->values[i]; // Set state of robot1
	}
}

void pcs::StateValidityChecker::updateStateVector(const ob::State *state, State q) {
	// cast the abstract state type to the type we expect
	const ob::RealVectorStateSpace::StateType *Q = state->as<ob::RealVectorStateSpace::StateType>();

	for (unsigned i = 0; i < n; i++) {
		Q->values[i] = q[i];
	}
}

void pcs::StateValidityChecker::printStateVector(const ob::State *state) {
	// cast the abstract state type to the type we expect
	const ob::RealVectorStateSpace::StateType *Q = state->as<ob::RealVectorStateSpace::StateType>();

	State q(n);

	for (unsigned i = 0; i < n; i++) {
		q[i] = Q->values[i]; // Set state of robot1
	}
	cout << "q: "; printVector(q);
}

State pcs::StateValidityChecker::sample_q() {
	State q(n);

	while (1) {
		// Randomly generate a chain
		for (int i = 0; i < n-1; i++) 
			q[i] = fRand(-PI_, PI_);
		q[n-1] = fRand(0, 2*PI_);

		int ik_sol = rand() % 2 + 1;
		if (IKproject(q, 0, ik_sol)) {
			break;
		}
	}

	// Check Constraints
	if (include_constraints && (!check_angles(q) || !obstacle_collision(q) || !self_collision(q)))  {
		q[0] = -1000;
		return q;
	}

	return q;
}

bool pcs::StateValidityChecker::IKproject(ob::State *state, int nc, int IK_sol, bool includeObs) {
	// nc - passive chain number
	State q(n), ik(m);

	retrieveStateVector(state, q);

	if (!IKproject(q, nc, IK_sol))
		return false;

	if (include_constraints && !check_angles(q))
		return false;

	// Check Constraints
	if (includeObs && include_constraints && (!obstacle_collision(q) || !self_collision(q)))
		return false;

	updateStateVector(state, q);

	return true;	
}

bool pcs::StateValidityChecker::IKproject(State &q, int nc, int IK_sol) {

	// nc - passive chain number

	State q_IK(3);
	State p_left(3), p_right(3), pose(3);

	if (nc < n-3) { // All passive chains except the last one
		if (nc==0)
			p_left = {0,0,0};
		else {
			FK_left(q, nc);
			p_left = get_FK_sol_left();
		}

		FK_right(q, n-3-nc);
		p_right = get_FK_sol_right();
		p_right[2] -= PI_;
		p_right[2] = fmod (p_right[2],  2*PI_);
		if (p_right[2] > PI_)
			p_right[2] -= 2*PI_;
		if (p_right[2] < -PI_)
			p_right[2] += 2*PI_;

		pose = {p_right[0]*cos(p_left[2]) - p_left[0]*cos(p_left[2]) - p_left[1]*sin(p_left[2]) + p_right[1]*sin(p_left[2]), p_right[1]*cos(p_left[2]) - p_left[1]*cos(p_left[2]) + p_left[0]*sin(p_left[2]) - p_right[0]*sin(p_left[2]), (p_right[2]-p_left[2])};
		if (pose[2] > PI_)
			pose[2] -= 2*PI_;
		if (pose[2] < -PI_)
			pose[2] += 2*PI_;

		if (IKp(pose, IK_sol, L))
			q_IK = get_IK_sol_q();
		else
			return false;

		q[nc] = q_IK[0];// + p_left[2];
		q[nc+1] = q_IK[1];
		q[nc+2] = q_IK[2];

	}
	if (nc == n-3)  { // The last passive chain (not including base) - special treatment
		FK_left_half(q, n-3);
		p_left = get_FK_sol_left();
		p_left[2] += PI_;
		p_left[2] = fmod (p_left[2],  2*PI_);
		if (p_left[2] > PI_)
			p_left[2] -= 2*PI_;
		if (p_left[2] < -PI_)
			p_left[2] += 2*PI_;

		pose = {p_left[0] - get_bx(), p_left[1] - get_by(), p_left[2]};
		if (pose[2] > PI_)
			pose[2] -= 2*PI_;
		if (pose[2] < -PI_)
			pose[2] += 2*PI_;

		if (IKp(pose, IK_sol, L))
			q_IK = get_IK_sol_q();
		else
			return false;

		/*if (q_IK[0] < 0)
			q_IK[0] += 2*PI_;*/

		q[n-1] = q_IK[0];
		q[n-2] = -q_IK[1];
		q[n-3] = -q_IK[2];

		if (q[n-1] < 0)	q[n-1] += 2*PI_;
		if (q[n-1] > 2*PI_) q[n-1] -= 2*PI_;
	}
	if (nc == n-2) { // Passive chain including the base and the right base joint
		p_left = {0,0,0};

		State qr(n-3);
		for (int i = n-2, j = 0; i > 1; i--, j++)
			qr[j] = -q[i];

		double x = L, y = 0, theta = 0, Lp;
		for (int i = 0; i < qr.size(); i++) {
			Lp = L;
			if (i==qr.size()-1)
				Lp = L/2;

			theta += qr[i];
			x += Lp*cos(theta);
			y += Lp*sin(theta);
		}

		p_right = {x, y, theta - PI_};
		p_right[2] = fmod (p_right[2],  2*PI_);
		if (p_right[2] > PI_)
			p_right[2] -= 2*PI_;
		if (p_right[2] < -PI_)
			p_right[2] += 2*PI_;

		pose = {p_right[0]*cos(p_left[2]) - p_left[0]*cos(p_left[2]) - p_left[1]*sin(p_left[2]) + p_right[1]*sin(p_left[2]), p_right[1]*cos(p_left[2]) - p_left[1]*cos(p_left[2]) + p_left[0]*sin(p_left[2]) - p_right[0]*sin(p_left[2]), (p_right[2]-p_left[2])};
		if (pose[2] > PI_)
			pose[2] -= 2*PI_;
		if (pose[2] < -PI_)
			pose[2] += 2*PI_;

		if (IKp(pose, IK_sol, get_b()))
			q_IK = get_IK_sol_q();
		else
			return false;

		double al = atan(get_by()/get_bx());

		q[n-1] = PI_ - q_IK[0] + al;
		q[0] = PI_ + q_IK[1] + al;
		q[1] = q_IK[2];

		if (q[0] < -PI_)	q[0] += 2*PI_;
		if (q[0] > PI_) q[0] -= 2*PI_;
		if (q[1] < -PI_)	q[1] += 2*PI_;
		if (q[1] > PI_) q[1] -= 2*PI_;
		if (q[n-1] < 0)	q[n-1] += 2*PI_;
		if (q[n-1] > 2*PI_) q[n-1] -= 2*PI_;
	}
	if (nc == n-1) { // Passive chain including the base and the left base joint

		State ql(n-3);
		for (int i = 1; i < n-2; i++)
			ql[i-1] = q[i];
		ql[0] = PI_ + ql[0];

		double x = 0, y = 0, theta = 0, Lp;
		for (int i = 0; i < ql.size(); i++) {
			Lp = L;
			if (i==ql.size()-1)
				Lp = L/2;

			theta += ql[i];
			x += Lp*cos(theta);
			y += Lp*sin(theta);
		}

		p_left = {x, y, theta + PI_};
		p_left[2] = fmod (p_left[2],  2*PI_);
		if (p_left[2] > PI_)
			p_left[2] -= 2*PI_;
		if (p_left[2] < -PI_)
			p_left[2] += 2*PI_;

		pose = {p_left[0] - L, p_left[1], p_left[2]};
		if (pose[2] > PI_)
			pose[2] -= 2*PI_;
		if (pose[2] < -PI_)
			pose[2] += 2*PI_;

		if (IKp(pose, IK_sol, get_b()))
			q_IK = get_IK_sol_q();
		else
			return false;

		double al = atan(get_by()/get_bx());

		q[0] = PI_ - q_IK[0] + al;
		q[n-1] = q_IK[1] + al;
		q[n-2] = -q_IK[2];

		if (q[0] < -PI_)	q[0] += 2*PI_;
		if (q[0] > PI_) q[0] -= 2*PI_;
		if (q[n-2] < -PI_)	q[n-2] += 2*PI_;
		if (q[n-2] > PI_) q[n-2] -= 2*PI_;
		if (q[n-1] < 0)	q[n-1] += 2*PI_;
		if (q[n-1] > 2*PI_) q[n-1] -= 2*PI_;
	}

	return true;
}

State pcs::StateValidityChecker::identify_state_ik(const ob::State *state) {

	State q(n);
	retrieveStateVector(state, q);

	return identify_state_ik(q);
}

State pcs::StateValidityChecker::identify_state_ik(State q) {

	State q_temp(n), ik(m, -1);

	double tol = 0.05;

	for (int nc = 0; nc < m; nc++) {
		for (int IK_sol = 1; IK_sol <= 2; IK_sol++) {
			q_temp = q;

			if (IKproject(q_temp, nc, IK_sol)) {
				VectorInt idx(3);
				if (nc < n-2)
					idx = {nc, nc+1, nc+2};
				else {
					if (nc == n-2)
						idx = {0, n-1, n-2};
					if (nc == n-1)
						idx = {n-1, 0, 1};
				}

				if (fabs(q_temp[idx[0]]-q[idx[0]]) < tol && fabs(q_temp[idx[1]]-q[idx[1]]) < tol && fabs(q_temp[idx[2]]-q[idx[2]]) < tol) {
					ik[nc] = IK_sol;
					break;
				}
			}
		}
		//if (ik[nc]==-1)
		//	cout << "Error: IK not found.\n";
	}

	return ik;
}
// ------------------- Validity check

// Validates a state by computing the passive chain based on a specific IK solution (input) and checking collision
bool pcs::StateValidityChecker::isValid(const ob::State *state, int active_chain, int IK_sol , bool project) {

	isValid_counter++;

	State q(n);
	retrieveStateVector(state, q);

	if (!IKproject(q, active_chain, IK_sol))
		return false;

	// Check Constraints
	if (include_constraints && (!check_angles(q) || !obstacle_collision(q) || !self_collision(q)))
		return false;

	if (project) {
		//ik[active_chain] = IK_sol;
		updateStateVector(state, q);//, ik);
	}

	return true;	
}

bool pcs::StateValidityChecker::checkMotion(const ob::State *s1, const ob::State *s2, int active_chain, int ik_sol)
{
	// We assume motion starts and ends in a valid configuration - due to projection
	bool result = true;
	State q1(n), q2(n);
	retrieveStateVector(s1,q1);
	retrieveStateVector(s2,q2);
	int nd = normDistance(q1,q2)/dq;
	//int nd = stateSpace_->validSegmentCount(s1, s2);

	// initialize the queue of test positions
	std::queue< std::pair<int, int> > pos;

	if (nd >= 2)
	{
		// temporary storage for the interpolated state
		ob::State *test = mysi_->allocState();

		if (0) { // serial search
			for (int i = 0; i <= nd; i++) {
				stateSpace_->interpolate(s1, s2, (double)i / (double)nd, test);	

				//if (IKproject(test, active_chain, ik_sol)) {
				if (!isValid(test, active_chain, ik_sol, false)) {
					result = false;
					break;
				}
			}
		}
		else { // Binary search
			pos.push(std::make_pair(1, nd - 1));

			// temporary storage for the checked state
			ob::State *test = mysi_->allocState();

			// repeatedly subdivide the path segment in the middle (and check the middle)
			while (!pos.empty())
			{
				std::pair<int, int> x = pos.front();

				int mid = (x.first + x.second) / 2;
				stateSpace_->interpolate(s1, s2, (double)mid / (double)nd, test);

				if (!isValid(test, active_chain, ik_sol, false))
				{
					result = false;
					break;
				}
				pos.pop();

				if (x.first < mid)
					pos.push(std::make_pair(x.first, mid - 1));
				if (x.second > mid)
					pos.push(std::make_pair(mid + 1, x.second));
			}
		}

		mysi_->freeState(test);
	}
	return result;
}

// ------------------------------------ RBS -------------------------------------------

// Validates a state by computing the passive chain based on a specific IK solution (input) and checking collision
bool pcs::StateValidityChecker::isValidRBS(State& q, int active_chain, int IK_sol) {

	isValid_counter++;

	if (!IKproject(q, active_chain, IK_sol))
		return false;

	// Check Constraints
	if (include_constraints && (!check_angles(q) || !obstacle_collision(q) || !self_collision(q)))
		return false;

	return true;
}

// Calls the Recursive Bi-Section algorithm (Hauser)
bool pcs::StateValidityChecker::checkMotionRBS(const ob::State *s1, const ob::State *s2, int active_chain, int ik_sol)
{
	// We assume motion starts and ends in a valid configuration - due to projection
	bool result = true;

	State q1(n), q2(n);
	retrieveStateVector(s1, q1);
	retrieveStateVector(s2, q2);

	result = checkMotionRBS(q1, q2, active_chain, ik_sol, 0, 0);

	return result;
}

// Implements local-connection using Recursive Bi-Section Technique (Hauser)
bool pcs::StateValidityChecker::checkMotionRBS(State q1, State q2, int active_chain, int ik_sol, int recursion_depth, int non_decrease_count) {

	// Check if reached the required resolution
	double d = normDistance(q1,q2); //normVector(angle_distance(q1, q2));
	if (d < RBS_tol)
		return true;

	if (recursion_depth > RBS_max_depth || non_decrease_count > 100)
		return false;

	State q_mid(n); //= midpoint(q1, q2);
	for (int i = 0; i < n; i++)
		q_mid[i] = (q1[i]+q2[i])/2;

	// Check obstacles collisions and joint limits
	if (!isValidRBS(q_mid, active_chain, ik_sol)) // Also updates s_mid with the projected value
		return false;

	if ( normVector(angle_distance(q1, q_mid)) > d || normVector(angle_distance(q_mid, q2)) > d)
		non_decrease_count++;

	if ( checkMotionRBS(q1, q_mid, active_chain, ik_sol, recursion_depth+1, non_decrease_count) && checkMotionRBS(q_mid, q2, active_chain, ik_sol, recursion_depth+1, non_decrease_count) )
		return true;
	else
		return false;
}

// *************** Reconstruct the RBS - for post-processing and validation

// Reconstruct local connection with the Recursive Bi-Section algorithm (Hauser)
bool pcs::StateValidityChecker::reconstructRBS(const ob::State *s1, const ob::State *s2, Matrix &Confs, int active_chain, int ik_sol)
{
	State q1(n), q2(n);
	retrieveStateVector(s1, q1);
	retrieveStateVector(s2, q2);

	Confs.push_back(q1);
	Confs.push_back(q2);

	return reconstructRBS(q1, q2, active_chain, ik_sol, Confs, 0, 1, 1, 0);
}

bool pcs::StateValidityChecker::reconstructRBS(State q1, State q2, int active_chain, int ik_sol, Matrix &M, int iteration, int last_index, int firstORsecond, int non_decrease_count) {

	// firstORsecond - tells if the iteration is from the first or second call for the recursion (in the last iteration).
	// last_index - the last index that was added to M.

	iteration++;

	// Check if reached the required resolution
	double d = normDistance(q1, q2);
	if (d < RBS_tol)
		return true;

	if (iteration > RBS_max_depth || non_decrease_count > 100)
		return false;

	State q_mid(n);
	for (int i = 0; i < n; i++)
		q_mid[i] = (q1[i]+q2[i])/2;

	// Check obstacles collisions and joint limits
	if (!isValidRBS(q_mid, active_chain, ik_sol)) // Also updates s_mid with the projected value
		return false;

	if ( normVector(angle_distance(q1, q_mid)) > d || normVector(angle_distance(q_mid, q2)) > d)
		non_decrease_count++;

	if (firstORsecond==1)
		M.insert(M.begin()+last_index, q_mid); // Inefficient operation, but this is only for post-processing and validation
	else
		M.insert(M.begin()+(++last_index), q_mid); // Inefficient operation, but this is only for post-processing and validation

	int prev_size = M.size();
	if (!reconstructRBS(q1, q_mid, active_chain, ik_sol, M, iteration, last_index, 1, non_decrease_count))
		return false;
	last_index += M.size()-prev_size;
	if (!reconstructRBS(q_mid, q2, active_chain, ik_sol, M, iteration, last_index, 2, non_decrease_count))
		return false;

	return true;
}

State pcs::StateValidityChecker::midpoint(State q1, State q2) {

	State q_mid(n);

	if (include_constraints)
		for (int i = 0; i < n; i++)
			q_mid[i] = (q1[i]+q2[i])/2;
	else {
		State dq = angle_distance(q1, q2);
		for (int i = 0; i < n; i++)
			q_mid[i] = q1[i] - dq[i]*0.5;
	}

	for (int i = 0; i < n; i++) {
		if (q_mid[i] > PI_)
			q_mid[i] -= 2*PI_;
		if (q_mid[i] < -PI_)
			q_mid[i] += 2*PI_;
	}
	if (q_mid[n-1] < 0)
		q_mid[n-1] += 2*PI_;

	return q_mid;
}

State pcs::StateValidityChecker::angle_distance(State q1, State q2) {

	State dq(n);

	for (int i = 0; i < n; i++)
		dq[i] = fmod((q1[i]-q2[i]) + PI_, 2*PI_) - PI_;

	return dq;

}

double pcs::StateValidityChecker::normDistance(State a1, State a2) {

	double sum = 0;
	for (int i=0; i < a1.size(); i++)
		sum += pow(a1[i]-a2[i], 2);
	return sqrt(sum);
}


double pcs::StateValidityChecker::normVector(State q) {

	double sum;
	for (int i = 0; i < n; i++)
		sum += q[i]*q[i];

	return sqrt(sum);
}

// ------------------------------- Constraints functions ---------------------------

bool pcs::StateValidityChecker::check_angles(State q, double f) {

	for (int i = 0; i < n-1; i++)
		if (q[i] > f*get_qminmax() || q[i] < -f*get_qminmax())
			return false;
	if (q[n-1] < 0)
		return false;

	return true;
}

bool pcs::StateValidityChecker::self_collision(State q, double f) {

	double Ax, Ay, Bx, By, Cx, Cy, Dx, Dy, l = getL();
	Ax = Ay = 0;

	double cum_q = 0, cum_q_CD;
	for (int i = 0; i < n-3; i++) {
		cum_q += q[i];

		Bx = Ax + l*cos(cum_q);
		By = Ay + l*sin(cum_q);

		cum_q_CD = cum_q + q[i+1];
		Cx = Bx + l*cos(cum_q_CD);
		Cy = By + l*sin(cum_q_CD);
		for (int j = i+2; j < n-1; j++) {
			cum_q_CD += q[j];
			Dx = Cx + l*cos(cum_q_CD);
			Dy = Cy + l*sin(cum_q_CD);

			// ---- Check if two lines intersect ---
			double s1_x = Bx - Ax;
			double s1_y = By - Ay;
			double s2_x = Dx - Cx;
			double s2_y = Dy - Cy;
			double s = (-s1_y * (Ax - Cx) + s1_x * (Ay - Cy)) / (-s2_x * s1_y + s1_x * s2_y);
			double t = ( s2_x * (Ay - Cy) - s2_y * (Ax - Cx)) / (-s2_x * s1_y + s1_x * s2_y);

			double minLim = -0.2 * f, maxLim = 1.2 * f;
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
bool pcs::StateValidityChecker::LinesIntersect(State A, State B, State C, State D, double f) {

	double s1_x, s1_y, s2_x, s2_y;
	s1_x = B[0] - A[0];
	s1_y = B[1] - A[1];
	s2_x = D[0] - C[0];
	s2_y = D[1] - C[1];

	double s, t;
	s = (-s1_y * (A[0] - C[0]) + s1_x * (A[1] - C[1])) / (-s2_x * s1_y + s1_x * s2_y);
	t = ( s2_x * (A[1] - C[1]) - s2_y * (A[0] - C[0])) / (-s2_x * s1_y + s1_x * s2_y);

	double minLim = -0.2*f, maxLim = 1.2*f;
	if (s >= minLim && s <= maxLim && t >= minLim && t <= maxLim)
		return false;

	return true; // No collision
}

bool pcs::StateValidityChecker::obstacle_collision(State q, double f) {

	double x, y, l = getL();
	x = y = 0;

	double cum_q = 0;
	for (int i = 0; i < n-1; i++) {
		cum_q += q[i];

		x = x + l*cos(cum_q);
		y = y + l*sin(cum_q);

		for (int j = 0; j < obs.size(); j++) {
			if ( (x-obs[j][0])*(x-obs[j][0]) + (y-obs[j][1])*(y-obs[j][1]) < (obs[j][2]+f*l)*(obs[j][2]+f*l) )
				return false;
		}
	}
	return true;
}

// -----------------------------------------------------------------------------

