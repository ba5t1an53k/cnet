#pragma once
#include <Eigen\Dense>
#include <iostream>
#include <cmath>
#include "activation_function.h"
/*
This class contains a vectorized implementation of a multi-layer perceptron neural network based on Eigen.
For implementation details please see:	http://peterroelants.github.io/posts/neural_network_implementation_part04/
										https://sudeepraja.github.io/Neural/
Note: This implementation takes a slightly different approach compared to both links.

Note: This implementation currently just supports one hidden layer.
*/
class neural_network
{
private:
	/*
	Hidden layer's activation function
	*/
	activation_function* _ha;
	/*
	Output layer's activation function
	*/
	activation_function* _oa;
	/*
	Column vector of input samples.
	*/
	Eigen::MatrixXd* _x;
	/*
	Weigth matrix from input to hidden layer with dimensions size(_h)*size(_x+1)
	*/
	Eigen::MatrixXd* _w0;
	/*
	Weigth matrix from hidden to output layer with dimensions size(_y)*size(_h+1)
	*/
	Eigen::MatrixXd* _w1;
	/*
	Column vector to store calculated values of the hidden layer
	*/
	Eigen::MatrixXd* _h;
	/*
	Column vector to store calculated values of the output layer
	*/
	Eigen::MatrixXd* _y;
	/*
	Hidden layer's bias term
	*/
	double _b0;
	/*
	Output layer's bias term
	*/
	double _b1;
	/*
	Softmax function (use for probability estimates)
	@in: x -> vector to apply probability estimates to. Note: Modifies x's values
	*/
	void softmax_function(Eigen::MatrixXd* x);
	/*
	Cross entropy loss function (use for classification)
	@in: y -> vector of outputs
	@in: t -> vector of targets
	@return: the cross entropy loss
	*/
	double cross_entropy_loss(Eigen::MatrixXd& y, Eigen::MatrixXd& t);
	/*
	Inverts sign of a negative x
	@in: x -> value to possibly invert sign
	@return: x if x >= 0, (-1)*x if x < 0
	*/
	double force_positive(double x);
	/*
	MSE of a given y and t
	@in: y -> output vector of prediction
	@in: t -> target vector
	Note: Buggy, do not use. :)
	*/
	double mean_square_error(Eigen::MatrixXd& y, Eigen::MatrixXd& t);
	/*
	Setup function for the network. See constructor for paramter details 
	*/
	void setup(int num_input, int num_hidden, int num_out);
	/*
	Performs the stochastic backpropagation algorithmn
	@in: alpha -> learning rate
	@in: residual -> difference between the predicted output and the expected target value
	*/
	void neural_network::back_prop(double alpha, Eigen::MatrixXd& residual);
	/*
	Calculates the delta for the output layer
	@in: output_derivative -> derivative of the output layer
	@in: residual -> residual at the output layer
	@out: output_delta -> delta of the output layer
	*/
	void neural_network::calc_output_delta(Eigen::MatrixXd& output_derivative, Eigen::MatrixXd& residual, Eigen::MatrixXd* output_delta);
	/*
	Calculates the delta of the hidden layer
	@in: hidden_derivative -> derivative of the hidden layer
	@in: hidden_weight -> weight matrix from previous to this layer
	@in: previous_delta -> delta of the previous layer
	@out : hidden_delta -> delta of this layer
	*/
	void neural_network::calc_hidden_delta(Eigen::MatrixXd& hidden_derivative, Eigen::MatrixXd& hidden_weight, Eigen::MatrixXd& previous_delta, Eigen::MatrixXd* hidden_delta);
	/*
	Updates the weight metrices
	@in: alpha -> learning rate fro gradient descent
	@in: hidden_gradient -> gradient of the hidden layer
	@in: out_gradient -> gradient of the output layer
	*/
	void update_weights(double alpha, Eigen::MatrixXd& hidden_gradient, Eigen::MatrixXd& out_gradient);

public:
	/*
	Contructor takes the amounts of input-, hidden- and output-neurons
	
	@in: num_input -> amount of input neurons
	@in: num_hidden -> amount of hidden neurons
	@in: num_output -> amount of output neurons
	*/
	explicit neural_network(int num_input, int num_hidden, int num_out);
	/*
	Deconstructor 
	*/
	~neural_network();
	/*
	Predicts the associated classes of a given sample
	
	@in: input -> the column vector containing the sample to predict
	@out: output -> the column vector containing class probabilities.
	*/
	void predict(Eigen::MatrixXd& input, Eigen::MatrixXd** output);
	/*
	Trains the network with specified parameters
	@in: sample_matrix -> matrix containing the training samples
	@in: target_matrix -> matrix containing the training targets
	@in: alpha -> the gradient descent learning rate
	@out: error -> the training error over all samples
	*/
	void train(Eigen::MatrixXd& sample_matrix, Eigen::MatrixXd& target_matrix, double alpha, double min_error_deviation, double* error);
	/*
	Calculates the residual between output and target
	@in: output -> network's prediction
	@in: target -> expected target
	@out: residual -> difference between target and predicted
	*/
	void neural_network::calc_residual(Eigen::MatrixXd& output, Eigen::MatrixXd& target, Eigen::MatrixXd* residual);
	
	/*
	Sets the activation function for the hidden layer (if not set, logistic function is used)

	Note: If pointer is passed, this class takes ownership of that passed instance
	
	*/
	void set_hidden_activation(activation_function* activation);
	/*
	Sets the activation for the output layer (if not set, logistic function is used)
	
	Note: If pointer is passed, this class takes ownership of that passed instance
	
	*/
	void set_output_activation(activation_function* activation);
	/*
	Logistic function
	*/
	double logistic_function(double x);
	/*
	Derivative of the Logistic function
	*/
	double logistic_derivative(double x);
};

