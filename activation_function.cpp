#include "stdafx.h"
#include "activation_function.h"

double logistic_activation::apply_function(double x) {
	return 1 / (1 + exp((-1) * x));
}

double logistic_activation::apply_derivative(double x) {
	return x*(1 - x);
}

double tanh_activation::apply_function(double x) {
	return (exp(x) - exp((-1)*x)) / (exp(x) + exp((-1)*x));
}

double tanh_activation::apply_derivative(double x) {
	return 1-pow(x, 2.0);
}

double relu_activation::apply_function(double x) {
	if (x < 0)
	{
		return 0;
	}
	return x;
}

double relu_activation::apply_derivative(double x) {
	if (x < 0)
	{
		return 0;
	}
	return 1;
}

