// neural_net.cpp : Definiert den Einstiegspunkt für die Konsolenanwendung.
//

#include "stdafx.h"
#include <iostream>
#include <Eigen\Dense>
#include "neural_network.h"

int main() {
	Eigen::MatrixXd in(4, 2);
	in(0, 0) = 0;
	in(0, 1) = 0;
	in(1, 0) = 1;
	in(1, 1) = 0;
	in(2, 0) = 1;
	in(2, 1) = 0;
	in(3, 0) = 1;
	in(3, 1) = 1;
	Eigen::MatrixXd target(4,1);
	target(0, 0) = 0;
	target(1, 0) = 1;
	target(2, 0) = 1;
	target(3, 0) = 0;
	Eigen::MatrixXd residual;
	neural_network net(2,2,1);
	double error = 0;
	net.train(in, target, 0.1, 0.00000001, &error);
	return 0;
}


