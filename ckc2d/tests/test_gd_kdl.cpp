#include "kdl_class.h"

#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#define ROBOTS_DISTANCE 900
#define ROD_LENGTH 300

double fRand(double fMin, double fMax)
{
	double f = (double)rand() / RAND_MAX;
	return fMin + f * (fMax - fMin);
}

double dist(State p1, State p2) {
	double sum = 0;
	for (int i = 0; i < p1.size(); i++)
		sum += (p1[i]-p2[i])*(p1[i]-p2[i]);

	return sqrt(sum);
}

int main() {

	int Seed = time(NULL);
	srand( Seed );
	cout << "Seed in testing: " << Seed << endl;

	int n = 5;

	// KDL
	kdl K(5, -1);

	int N = 1e6;
	State q(n), qp_kdl(n);
	double kdl_time = 0;

	for (int i = 0; i < N; i++) {

		for (int j = 0; j < n-1; j++)
			q[j] = fRand(-PI, PI);
		q[n-1] = fRand(0, 2*PI);

		// KDL
		clock_t begin = clock();
		bool Ksuc = K.GD(q);
		qp_kdl = K.get_GD_result();
		kdl_time += double(clock() - begin) / CLOCKS_PER_SEC;

	}

    cout << "Avg. time: " << kdl_time / N << "sec." << endl;


}

