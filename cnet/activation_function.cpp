#include "stdafx.h"
#include "activation_function.h"

double logistic_activation::apply_function(double x) {
	return 1 / (1 + exp((-1)*x));
}

double logistic_activation::apply_derivative(double x) {
	return x*(1 - x);
}

double tanh_activation::apply_function(double x) {
	return (2 / (1 + exp((-1)*(2*x))))-1;
}

double tanh_activation::apply_derivative(double x) {
	return x*(1 - x);
}

double relu_activation::apply_function(double x) {
	return std::fmax(0,x);
}

double relu_activation::apply_derivative(double x) {
	if (x > 0)
	{
		return 1;
	}
	return 0;
}

