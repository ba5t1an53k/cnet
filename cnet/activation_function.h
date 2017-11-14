#pragma once
#include <cmath>
/*
This class defines an interface for activation functions used as non-linear term in neural network's approximated function
*/
class activation_function
{
public:
	/*
	Applies the specific function to a given value x
	@in: x -> function's input value
	@return: the calculated y of f(x), y=f(x)
	*/
	virtual double apply_function(double x) = 0;
	/*
	Applies the specific derivative of the activation function to a given value x
	@in: x -> derivatives's input value
	@return: the calculated y of f'(x), y=f'(x)
	*/
	virtual double apply_derivative(double x) = 0;
};

/*
This class implements the logistic function and it's derivative
*/
class logistic_activation : public activation_function
{
public:
	/*
	Logistic function
	See interface activation_function for parameter information
	*/
	double activation_function::apply_function(double x) override;
	/*
	Logistic function's derivative
	See interface activation_function for parameter information
	*/
	double activation_function::apply_derivative(double x) override;
};
/*
This class implements the tangens hyperbolicus (tanh) and it's derivative
*/
class tanh_activation : public activation_function
{
public:
	/*
	Tangens Hyperbolicus 
	See interface activation_function for parameter information
	*/
	double activation_function::apply_function(double x) override;
	/*
	Tangens Hyperbolicus' derivative
	See interface activation_function for parameter information
	*/
	double activation_function::apply_derivative(double x) override;
};
/*
This class implements the linear rectifier (ReLu) and it's derivative
*/
class relu_activation : public activation_function
{
public:
	/*
	Linear Rectifier
	See interface activation_function for parameter information
	*/
	double activation_function::apply_function(double x) override;
	/*
	Linear Rectifier's derivative
	See interface activation_function for parameter information
	*/
	double activation_function::apply_derivative(double x) override;
};

/*
This class implements the leaky ReLu function and it's derivative
*/
class softplus_activation : public activation_function
{
public:
	/*
	Leaky ReLu
	See interface activation_function for parameter information
	*/
	double activation_function::apply_function(double x) override;
	/*
	Leaky ReLu's derivative
	See interface activation_function for parameter information
	*/
	double activation_function::apply_derivative(double x) override;
};

/*
This class implements the Softplus function and it's derivative
*/
class leaky_relu_activation : public activation_function
{
public:
	/*
	Softplus 
	See interface activation_function for parameter information
	*/
	double activation_function::apply_function(double x) override;
	/*
	Softplus's derivative
	See interface activation_function for parameter information
	*/
	double activation_function::apply_derivative(double x) override;
};

