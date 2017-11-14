#include "stdafx.h"
#include "activation_function.h"

double logistic_activation::apply_function(double x) {
	return 1 / (1 + exp((-1)*x));
}

double logistic_activation::apply_derivative(double x) {
	return x*(1 - x);
}

