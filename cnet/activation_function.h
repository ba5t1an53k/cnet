#pragma once
#include <cmath>

class activation_function
{
public:
	virtual double apply_function(double x) = 0;
	virtual double apply_derivative(double x) = 0;
};

class logistic_activation : public activation_function
{
public:
	double activation_function::apply_function(double x) override;
	double activation_function::apply_derivative(double x) override;
};

class tanh_activation : public activation_function
{
public:
	double activation_function::apply_function(double x) override;
	double activation_function::apply_derivative(double x) override;
};

