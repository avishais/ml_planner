#include "kdl_class.h"

// Constructor for the robots
kdl::kdl(int joints_num, double custom_num) {
	double l;

	if (joints_num == 20) {
		bx = 7; // Base for scenario with obs (n = 20)
		by = 4;
		l = 1;

	}
	else {
		bx = 3.2;//4;//
		by = 0;
		l = 1;
	}

	if (custom_num==-1)
		custom_num = 0.3;

	double total_length = 6.5;
	double base_links_ratio = 0.3;

	// l = total_length/((joints_num-1)*(1+base_links_ratio)); // Link length.
	// bx = base_links_ratio*(l*(joints_num-1));
	// by = 0;

	L.resize(joints_num-1);
	// L = {1, 3, 7};
	for (int i = 0; i < joints_num-1; i++)
		L[i] = l;

	n = joints_num;

	// Joint limits
	qminmax = 179.9/180*PI_;

	initMatrix(T_pose, 4, 4);
	T_pose = {{1, 0, 0, bx}, {0, 1, 0, by}, {0, 0, 1, 0}, {0, 0, 0, 1}};

	for (int i = 0; i < 3; i++) {
		cartposIK.p(i) = T_pose[i][3];
		for (int j = 0; j < 3; j++)
			cartposIK.M(i,j) = T_pose[i][j];
	}

	//Definition of a kinematic chain & add segments to the chain
		for (int i = 0; i < joints_num-1; i++)
		chain.addSegment(Segment(Joint(Joint::RotZ),Frame(Vector(L[i],0.0,0.0)))); 
	chain.addSegment(Segment(Joint(Joint::RotZ),Frame(Vector(0.0,0.0,0.0)))); 

	// Create joint array
	unsigned int nj = chain.getNrOfJoints();
	jointpositions = JntArray(nj);

	initVector(q_solution, nj);

	IK_time = 0;

	cout << "Initiated chain with " << nj << " joints.\n";
}

// ----- Descend -------

bool kdl::GD(State q_init) {

	bool valid = true;

	// Flip robot two vector
	State q(n);

	IK_counter++;
	clock_t begin = clock();

	// KDL
	ChainFkSolverPos_recursive fksolver = ChainFkSolverPos_recursive(chain); 	// Create solver based on kinematic chain
	ChainIkSolverVel_pinv iksolverv(chain);//Inverse velocity solver
	ChainIkSolverPos_NR iksolver(chain,fksolver,iksolverv,10000,1e-5);//Maximum 10000 iterations, stop at accuracy 1e-5

	//Creation of jntarrays:
	JntArray qKDL(chain.getNrOfJoints());
	JntArray qInit(chain.getNrOfJoints());

	double scale = 0.00001;
	for (int i = 0; i < chain.getNrOfJoints(); i++) {
		q_init[i] = floor(q_init[i] / scale + 0.5) * scale; // Chop accuracy due to bug in KDL
		qInit(i) = q_init[i];
	}

	// This is a fix since the last joint in KDL terms is the extension of the arm and is relative to the last link
	qInit(n-1) = PI_ - qInit(n-1);

	//Set destination frame
	KDL::Frame F_dest = cartposIK;//Frame(Vector(1.0, 1.0, 0.0));
	int ret = iksolver.CartToJnt(qInit, F_dest, qKDL);

	bool result = false;
	if (ret >= 0) {

		// Revert fix from above
		qKDL(n-1) = PI_ - qKDL(n-1);

		for (int i = 0; i < n; i++)
			if (fabs(qKDL(i)) < 1e-4)
				q[i] = 0;
			else
				q[i] = qKDL(i);

		for (int i = 0; i < q.size()-1; i++) {
			q[i] = fmod(q[i], 2*PI_);
			if (q[i]>PI_)
				q[i] -= 2*PI_;
			if (q[i]<-PI_)
				q[i] += 2*PI_;
		}
		q[n-1] = fmod (q[n-1],  2*PI_);
		if (q[n-1] > 2*PI_)
			q[n-1] -= 2*PI_;
		if (q[n-1] < 0)
			q[n-1] += 2*PI_;

		for (int i = 0; i < n; i++)
			q_solution[i] = q[i];

		result = true;
	}

	clock_t end = clock();
	IK_time += double(end - begin) / CLOCKS_PER_SEC;

	return result;
}

bool kdl::check_angle_limits(State q) {

	for (int i = 0; i < n; i++)
		if (fabs(q[i]) > qminmax)
			return false;
	return true;
}

State kdl::get_GD_result() {
	return q_solution;
}

// -----FK-------

// This is only for validation. There is no use for this function in terms of closed chain kinematics
void kdl::FK(State q) {

	// Create solver based on kinematic chain
	ChainFkSolverPos_recursive fksolver = ChainFkSolverPos_recursive(chain);

	for (int i = 0; i < q.size(); i++)
		jointpositions(i) = q[i];

	// Calculate forward position kinematics
	bool kinematics_status;
	kinematics_status = fksolver.JntToCart(jointpositions, cartposFK);

	initMatrix(T_fk, 4, 4);
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

State kdl::constraint(State q) {

	// KDL fix
	q[n-1] = PI_ - q[n-1];

	FK(q);
	Matrix T = get_FK_solution();

	State C = {T[0][3], T[1][3], atan2(T[1][0], T[0][0])};

	C[0] -= bx;
	C[1] -= by;

	return C;
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
	return deg * PI_ / 180.0;
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

void kdl::log_q(State q, bool New) {

	// Log env. info
	std::ofstream mf;
	mf.open("../paths/path_info.txt");
	mf << n << endl << 0 << endl << L[0] << endl << bx << endl << by << endl << qminmax << endl;
	mf.close();

	// New=true, erase and write new file
	std::ofstream myfile;

	if (New) {
		myfile.open("../paths/path.txt");
		myfile << 1 << endl;
	}
	else
		myfile.open("../paths/path.txt", ios::app);

	for (int i = 0; i < q.size(); i++)
		myfile << q[i] << " ";
	myfile << endl;

	myfile.close();
}

State kdl::rand_q() {
	State q(n);
	for (int i = 0; i < q.size(); i++)
		q[i] = -PI + (double)rand()/RAND_MAX * 2*PI;

	return q;
}
