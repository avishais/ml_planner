#include "kdl_class.h"

// Constructor for the robots
kdl::kdl() : L(ROD_LENGTH) {
	b = 166; // Height of base
	l1 = 124; // Distance from top facet of base to axis of link 1.
	l2 = 270; // Length between axes of link 2.
	l3a = 70; // Shortest distance from top axis of link 2 to axis of link 3. (possibly 69.96mm).
	l3b = 149.6; // Horizontal distance(in home position) from top axis of link 2 mutual face of link 3 with link 4.
	l4 = 152.4; // Distance from mutual face of link 3 and link 4 to axis of links 4 - 5.
	l5 = 72-13/2; // Distance from axis of link 5 to its facet.
	lee = 60+13/2; // Distance from facet of link 5 to EE point

	// Joint limits
	q1minmax = deg2rad(165);
	q2minmax = deg2rad(110);
	q3min = deg2rad(-110);
	q3max = deg2rad(70);
	q4minmax = deg2rad(160);
	q5minmax = deg2rad(120);
	q6minmax = deg2rad(400);

	initMatrix(T_fk, 4, 4);

	initMatrix(T_pose, 4, 4);
	T_pose = {{1, 0, 0, ROBOTS_DISTANCE}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};

	//Definition of a kinematic chain & add segments to the chain
	// Robot 1
	chain.addSegment(Segment(Joint(Joint::None),Frame(Vector(0.0,0.0,b)))); // Base link
	chain.addSegment(Segment(Joint(Joint::RotZ),Frame(Vector(0.0,0.0,l1)))); // Link 1
	chain.addSegment(Segment(Joint(Joint::RotY),Frame(Vector(0.0,0.0,l2)))); // Link 2
	chain.addSegment(Segment(Joint(Joint::RotY),Frame(Vector(l3b,0.0,l3a)))); // Link 3
	chain.addSegment(Segment(Joint(Joint::RotX),Frame(Vector(l4,0.0,0.0)))); // Link 4
	chain.addSegment(Segment(Joint(Joint::RotY),Frame(Vector(l5,0.0,0.0)))); // Link 5
	chain.addSegment(Segment(Joint(Joint::RotX),Frame(Vector(2*lee+L,0.0,0.0)))); // Rod + 2 EE
	// Robot 2 - backwards
	chain.addSegment(Segment(Joint(Joint::RotX),Frame(Vector(l5,0.0,0.0)))); // Link 5
	chain.addSegment(Segment(Joint(Joint::RotY),Frame(Vector(l4,0.0,0.0)))); // Link 4
	chain.addSegment(Segment(Joint(Joint::RotX),Frame(Vector(l3b,0.0,-l3a)))); // Link 3 "-"
	chain.addSegment(Segment(Joint(Joint::RotY),Frame(Vector(0.0,0.0,-l2)))); // Link 2
	chain.addSegment(Segment(Joint(Joint::RotY),Frame(Vector(0.0,0.0,-l1)))); // Link 1
	chain.addSegment(Segment(Joint(Joint::RotZ),Frame(Vector(0.0,0.0,-b)))); // Base link

	// Create joint array
	unsigned int nj = chain.getNrOfJoints();
	jointpositions = JntArray(nj);

	// Joint limits for KDL
	q_min = JntArray(nj);
	q_max = JntArray(nj);
	State q_min_vec = {-q1minmax, -q2minmax, q3min, -q4minmax, -q5minmax, -q6minmax, -q6minmax, -q5minmax, -q4minmax, q3min, -q2minmax, -q1minmax};
	State q_max_vec = {q1minmax, q2minmax, q3max, q4minmax, q5minmax, q6minmax, q6minmax, q5minmax, q4minmax, q3max, q2minmax, q1minmax};
	for (int i = 0; i < nj; i++) {
		q_min(i) = q_min_vec[i];
		q_max(i) = q_max_vec[i];
	}

	initVector(q_solution, nj);

	cout << "Initiated chain with " << nj << " joints.\n";
}

// ----- Descend -------

bool kdl::GD(State q_init_flip) {

	bool valid = true;

	// Flip robot two vector
	State q_init(q_init_flip.size());
	for (int i = 0; i < 6; i++)
		q_init[i] = q_init_flip[i];
	for (int i = 11, j = 6; i >= 6; i--,j++)
		q_init[j] = q_init_flip[i];
	q_init[11] = -q_init[11];

	State q(12);

	IK_counter++;
	auto begin = Clock::now();

	for (int i = 0; i < 3; i++) {
		cartposIK.p(i) = T_pose[i][3];
		for (int j = 0; j < 3; j++)
			cartposIK.M(i,j) = T_pose[i][j];
	}

	// KDL
	ChainFkSolverPos_recursive fksolver = ChainFkSolverPos_recursive(chain); 	// Create solver based on kinematic chain
	ChainIkSolverVel_pinv iksolverv(chain);//Inverse velocity solver
	ChainIkSolverPos_NR iksolver(chain,fksolver,iksolverv,10000,1e-5);//Maximum 10000 iterations, stop at accuracy 1e-5

	//Creation of jntarrays:
	JntArray qKDL(chain.getNrOfJoints());
	JntArray qInit(chain.getNrOfJoints());

	for (int i = 0; i < chain.getNrOfJoints(); i++)
		qInit(i) = q_init[i];

	//Set destination frame
	KDL::Frame F_dest = cartposIK;//Frame(Vector(1.0, 1.0, 0.0));
	int ret = iksolver.CartToJnt(qInit, F_dest, qKDL);

	bool result = false;
	if (ret >= 0) {

		for (int i = 0; i < 12; i++)
			if (fabs(qKDL(i)) < 1e-4)
				q[i] = 0;
			else
				q[i] = qKDL(i);

		for (int i = 0; i < q.size(); i++) {
			q[i] = fmod(q[i], 2*PI);
			if (q[i]>PI)
				q[i] -= 2*PI;
			if (q[i]<-PI)
				q[i] += 2*PI;
		}

		for (int i = 0; i < 6; i++)
			q_solution[i] = q[i];
		for (int i = 11, j = 6; i >= 6; i--,j++)
			q_solution[j] = q[i];
		q_solution[6] = -q_solution[6];

		if (include_joint_limits && !check_angle_limits(q_solution))
			result = false;
		else
			result = true;
	}

	IK_time += std::chrono::duration<double>(Clock::now() - begin).count();

	return result;
}

bool kdl::check_angle_limits(State q) {

	if (fabs(q[0]) > q1minmax)
		return false;
	if (fabs(q[1]) > q2minmax)
		return false;
	if (q[2] < q3min || q[2] > q3max)
		return false;
	if (fabs(q[3]) > q4minmax)
		return false;
	if (fabs(q[4]) > q5minmax)
		return false;
	if (fabs(q[5]) > q6minmax)
		return false;

	if (fabs(q[6]) > q1minmax)
		return false;
	if (fabs(q[7]) > q2minmax)
		return false;
	if (q[8] < q3min || q[8] > q3max)
		return false;
	if (fabs(q[9]) > q4minmax)
		return false;
	if (fabs(q[10]) > q5minmax)
		return false;
	if (fabs(q[11]) > q6minmax)
		return false;

	return true;
}

State kdl::get_GD_result() {
	return q_solution;
}

// -----FK-------

// This is only for validation. There is no use for this function in terms of closed chain kinematics
void kdl::FK(State q) {

	// Flip robot two vector
	State q_flip(q.size());
	for (int i = 0; i < 6; i++)
		q_flip[i] = q[i];
	for (int i = 11, j = 6; i >= 6; i--,j++)
		q_flip[j] = q[i];
	q_flip[11] = -q_flip[11];

	// Create solver based on kinematic chain
	ChainFkSolverPos_recursive fksolver = ChainFkSolverPos_recursive(chain);

	for (int i = 0; i < q_flip.size(); i++)
		jointpositions(i) = q_flip[i];

	// Calculate forward position kinematics
	bool kinematics_status;
	kinematics_status = fksolver.JntToCart(jointpositions, cartposFK);

	for (int i = 0; i < 3; i++) {
		T_fk[i][3] = cartposFK.p(i);
		for (int j = 0; j < 3; j++)
			T_fk[i][j] = cartposFK.M(i,j);
	}
	T_fk[3][0] = T_fk[3][1] = T_fk[3][2] = 0;
	T_fk[3][3] = 1;
}

Matrix kdl::get_FK_solution() {

	//printMatrix(T_fk_solution_1);

	return T_fk;
}

//------------------------

// Misc
void kdl::initMatrix(Matrix &M, int n, int m) {
	M.resize(n);
	for (int i = 0; i < n; ++i)
		M[i].resize(m);
}

void kdl::initVector(State &V, int n) {
	V.resize(n);
}

double kdl::deg2rad(double deg) {
	return deg * PI / 180.0;
}

void kdl::printMatrix(Matrix M) {
	for (unsigned i = 0; i < M.size(); i++) {
		for (unsigned j = 0; j < M[i].size(); j++)
			cout << M[i][j] << " ";
		cout << endl;
	}
}

void kdl::printVector(State p) {
	cout << "[";
	for (unsigned i = 0; i < p.size(); i++)
		cout << p[i] << " ";
	cout << "]" << endl;
}


void kdl::clearMatrix(Matrix &M) {
	for (unsigned i=0; i<M.size(); ++i)
		for (unsigned j=0; j<M[i].size(); ++j)
			M[i][j] = 0;;
}

void kdl::log_q(State q) {
	std::ofstream myfile;
	myfile.open("../paths/path.txt");

	myfile << 1 << endl;

	for (int i = 0; i < q.size(); i++)
		myfile << q[i] << " ";
	myfile << endl;

	myfile.close();
}

State kdl::rand_q() {
	State q(12);
	for (int i = 0; i < q.size(); i++)
		q[i] = -PI + (double)rand()/RAND_MAX * 2*PI;

	return q;
}
