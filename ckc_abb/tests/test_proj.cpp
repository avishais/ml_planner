#include "kdl_class.h"
#include "net_proj.h"

#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#include <chrono>  // for high_resolution_clock
typedef std::chrono::high_resolution_clock Clock;

#define ROBOTS_DISTANCE 900
#define ROD_LENGTH 300

double fRand(double fMin, double fMax)
{
	double f = (double)rand() / RAND_MAX;
	return fMin + f * (fMax - fMin);
}

int main() {

	int Seed = time(NULL);
	srand( Seed );
	cout << "Seed in testing: " << Seed << endl;

	// KDL
	kdl K;

	int n = 12;

	int N = 1e4;
	State q(n), qp_kdl(n);
	double kdl_time = 0;

	for (int i = 0; i < N; i++) {

		for (int j = 0; j < n-1; j++)
			q[j] = fRand(-PI, PI);
		q[n-1] = fRand(0, 2*PI);

		// KDL
		auto begin = Clock::now();
		bool Ksuc = K.GD(q);
		qp_kdl = K.get_GD_result();
		kdl_time += std::chrono::duration<double>(Clock::now() - begin).count();

		// K.log_q(qp_kdl);

	}

	net_proj U;
    double t = 0;

	State z(6, 0);
	State zmax = {1., 0.99441636, 0.59342104, 0.6667417,  0.61381936, 0.8222987};

    for (int i = 0; i < N; i++) {
		for (int j = 0; j < 6; j++)
			z[j] = fRand(0, zmax[j]);

        auto begin = Clock::now();
        U.decoder(z);
        t += std::chrono::duration<double>(Clock::now() - begin).count();
    }

	// q = U.decoder(z);
	// K.printVector(q);

	// K.log_q(q);

	// K.printVector(qp_kdl);
	// K.log_q(qp_kdl);
	// cin.ignore();
	// State qq = U.decoder(U.encoder(qp_kdl));
	// K.printVector(qq);
	// K.log_q(qq);

    cout << "Avg. kdl time: " << kdl_time / N << "sec." << endl;
	cout << "Avg. time: " << t / n << "sec." << endl;


}

