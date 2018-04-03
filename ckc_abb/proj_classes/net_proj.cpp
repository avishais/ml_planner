

#include "net_proj.h"

net_proj::net_proj() {
    import_weights();
}

void net_proj::import_weights() {
    const char* net_file = "../../tensorflow/net_abb3.netxt";
    double te;
    int r, w;

    ifstream F;
    F.open(net_file);

    F >> n;
    W.resize(n);
    b.resize(n);

    n_encoded = n/2;
    n_decoded = n - n_encoded;

    // Weights
    for (int i=0; i < n; i++) {
        F >> r >> w;
        W[i].resize(r,w);
        for (int j = 0; j < r; j++) {
            for (int k = 0; k < w; k++) {
                F >> W[i](j,k);
            }
        }
    }

    // Biases
    for (int i=0; i < n; i++) {
        F >> r;
        b[i].resize(r);
        for (int j = 0; j < r; j++) {
            F >> b[i](j);
        }
    }

    F >> x_max >> x_min;
}

VectorXd net_proj::normalize(VectorXd x){
    for (int i = 0; i < x.size(); i++)
        x(i) = (x(i) - x_min) / (x_max-x_min);
        return x;
}
	
    
VectorXd net_proj::denormalize(VectorXd x) {
    for (int i = 0; i < x.size(); i++)
        x(i) =  x(i) * (x_max-x_min) + x_min;
    return x;
}

// -------------------------------------------------------------

// Sigmoid function
double net_proj::sigmoid(double x) {
    // return x / (1 + fabs(x)); // Approximated
    return 1 / (1 + exp(-x)); // Definition
}

VectorXd net_proj::activation(VectorXd x) {
    for (int i = 0; i < x.size(); i++)
        // x(i) = tanh(x(i));
        x(i) = sigmoid(x(i));
    return x;
}

State net_proj::encoder(State x_in) {
    VectorXd x(x_in.size());
    for (int i = 0; i < x_in.size(); i++)
        x(i) = x_in[i];

    x = normalize(x);

    for (int i = 0; i < n_encoded; i++) 
        x = activation(x.transpose()*W[i] + b[i].transpose());

    State z(x.size());
    for (int i = 0; i < x.size(); i++)
        z[i] = x(i);
    
    return z;
}

State net_proj::decoder(State z) {
    VectorXd x(z.size());
    for (int i = 0; i < z.size(); i++)
        x(i) = z[i];

    for (int i = n_encoded; i < n_encoded+n_decoded; i++) 
        x = activation(x.transpose()*W[i] + b[i].transpose());

    x = denormalize(x);

    State x_out(x.size());
    for (int i = 0; i < x.size(); i++)
        x_out[i] = x(i);
    
    return x_out;
}

// -------------------------------------------------------------

// void net_proj::printWeight(Matrix M) {
//     cout << "[";
// 	for (unsigned i = 0; i < M.size(); i++) {
// 		for (unsigned j = 0; j < M[i].size(); j++)
// 			cout << M[i][j] << " ";
// 		i==M.size()-1 ? cout << "]" : cout << ""; 
//         cout << endl;
// 	}
// } 

// void net_proj::printBias(State M) {
//     cout << "[";
// 	for (unsigned i = 0; i < M.size(); i++) 
// 			cout << M[i] << " ";
//     cout << "]" << endl;
// } 

void net_proj::printWeight(MatrixXd M) {
    cout << "[";
	for (unsigned i = 0; i < M.rows(); i++) {
		for (unsigned j = 0; j < M.cols(); j++)
			cout << M(i,j) << " ";
		i==M.rows()-1 ? cout << "]" : cout << ""; 
        cout << endl;
	}
} 

void net_proj::printBias(VectorXd M) {
    cout << "[";
	for (unsigned i = 0; i < M.size(); i++) 
			cout << M(i) << " ";
    cout << "]" << endl;
} 

// int main() {
//     net_proj N;
//     double t = 0;
//     int n = 1e6;

//     State x(2);
//     for (int i = 0; i < n; i++) {
//         x[0] = rand() * 2 - 1;
//         x[1] = rand() * 2 - 1;

//         clock_t st = clock();
//         N.decoder(x);
//         t += double(clock() - st) / CLOCKS_PER_SEC;
//     }

//     cout << "Avg. time: " << t / n << "sec." << endl;
    

//     return 0;
// }


