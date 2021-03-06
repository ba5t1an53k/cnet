// neural_net.cpp : Definiert den Einstiegspunkt für die Konsolenanwendung.
//
#include "stdafx.h"
#include <iostream>
#include <Eigen\Dense>
#include "neural_network.h"
#include "normalizer.h"
#include "data_source.h"

int main() {
	Eigen::MatrixXd* data;
	Eigen::MatrixXd* target;
	data_source source("c:\\dev\\xor.txt");
	source.get_data_matrix(&data, &target);
	std::cout << "INFO::\n"<<*data << "\n"<< *target<<  std::endl;
	normalizer norm(data, -1, 1);
	norm.normalize_matrix();
	std::cout << *data << std::endl;
	std::cin.get();
	neural_network net(2, 2, 1);
	double error = 0;
	net.set_hidden_activation(new tanh_activation);
	net.set_output_activation(new tanh_activation);
	//net.set_hidden_activation(new relu_activation);
	net.train(*data, *target, 0.9, 0.01, 100000, &error);
	std::cin.get();
	return 0;
}


