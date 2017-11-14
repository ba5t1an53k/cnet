#include "stdafx.h"
#include "neural_network.h"

neural_network::neural_network(int num_input, int num_hidden, int num_out)
{
	this->setup(num_input, num_hidden, num_out);
}

double neural_network::mean_square_error(Eigen::MatrixXd & y, Eigen::MatrixXd & t)
{
	Eigen::MatrixXd temp = (t - y).unaryExpr([=](double x)-> double {
		return pow(x, 2.0);
	});
	return (1 / (double) temp.rows())*temp.sum();
}

void neural_network::setup(int num_input, int num_hidden, int num_out)
{
	this->_x = new Eigen::MatrixXd(num_input, 1);
	this->_h = new Eigen::MatrixXd(num_hidden, 1);
	this->_y = new Eigen::MatrixXd(num_out, 1);
	this->_w0 = new Eigen::MatrixXd();
	*this->_w0 = Eigen::MatrixXd::Random(num_hidden, num_input);
	this->_w1 = new Eigen::MatrixXd();
	*this->_w1 = Eigen::MatrixXd::Random(num_out, num_hidden);
	this->_b0 = 1;
	this->_b1 = 1;
	//Assuming set activation for specific layers will not be called
	_ha = new logistic_activation();
	_oa = new logistic_activation();

}

neural_network::~neural_network()
{
	if (_x)
	{
		delete _x;
	}
	if (_y)
	{
		delete _y;
	}
	if (_w0)
	{
		delete _w0;
	}
	if (_w1)
	{
		delete _w1;
	}
	if (_ha) {
		delete _ha;
	}
	if (_oa) {
		delete _oa;
	}
}

void neural_network::predict(Eigen::MatrixXd& values, Eigen::MatrixXd** output) {
	_x = &values;
	Eigen::MatrixXd temp = (*_w0)*(*_x);
	*_h = temp.unaryExpr([&](double x) -> double {
		x += _b0;
		return _ha->apply_function(x);
	});
	temp = (*_w1)*(*_h);
	*_y = temp.unaryExpr([&](double x) -> double {
		x += _b1;
		return _oa->apply_function(x);
	});
	*output = _y;
}

void neural_network::train(Eigen::MatrixXd & sample_matrix, Eigen::MatrixXd & target_matrix, double alpha, double min_error_deviation, double * error)
{
	int num_epochs = 0;
	bool is_training = true;
	double last_error = std::numeric_limits<double>::max();
	Eigen::MatrixXd* current_output = new Eigen::MatrixXd();
	do {
		Eigen::MatrixXd predictions(sample_matrix.rows(), target_matrix.cols());
		for (int i = 0; i < sample_matrix.rows(); i++) {
			
			Eigen::MatrixXd training_row = sample_matrix.row(i).transpose();
			Eigen::MatrixXd target_row = target_matrix.row(i).transpose();
			predict(training_row, &current_output);
			predictions.row(i) = current_output->transpose();
			Eigen::MatrixXd residual;
			calc_residual(*current_output, target_row, &residual);
			back_prop(alpha, residual);
		}
		double loss = mean_square_error(predictions, target_matrix);
		std::cout << "INFO:: Training epoch " << num_epochs++ << ", loss=" << std::to_string(loss) << std::endl;
		if ((last_error - loss) < min_error_deviation)
		{
			is_training = false;
		}

	} while (is_training);
	delete current_output;
}

void neural_network::calc_residual(Eigen::MatrixXd& output, Eigen::MatrixXd& target, Eigen::MatrixXd* residual) {
	*residual = output - target;
}

void neural_network::set_hidden_activation(activation_function * activation)
{
	if (_ha) {
		delete _ha;
	}
	this->_ha = activation;
}

void neural_network::set_output_activation(activation_function * activation)
{
	if (_oa) {
		delete _oa;
	}
	this->_oa = activation;
}

void neural_network::calc_output_delta(Eigen::MatrixXd& output_derivative, Eigen::MatrixXd& residual, Eigen::MatrixXd* output_delta) {
	*output_delta = output_derivative.cwiseProduct(residual);
}

void neural_network::calc_hidden_delta(Eigen::MatrixXd& hidden_derivative, Eigen::MatrixXd& hidden_weight, Eigen::MatrixXd& previous_delta, Eigen::MatrixXd* hidden_delta) {
	Eigen::MatrixXd temp = hidden_weight.transpose()*previous_delta;
	*hidden_delta = hidden_derivative.cwiseProduct(temp);
}

void neural_network::update_weights(double alpha, Eigen::MatrixXd& hidden_gradient, Eigen::MatrixXd& out_gradient) {
	*_w0 -= (alpha*hidden_gradient);
	*_w1 -= (alpha*out_gradient);	
}

void neural_network::back_prop(double alpha, Eigen::MatrixXd& residual) {
	Eigen::MatrixXd out_delta, hid_delta;
	//apply derivative of logistic function x*(x-1) to the output
	Eigen::MatrixXd output_derivative = _y->unaryExpr([=](double x)->double {
		return _oa->apply_derivative(x);
	});
	//calculate output delta
	calc_output_delta(output_derivative, residual, &out_delta);
	//calculate output gradient
	Eigen::MatrixXd output_gradient = out_delta*_h->transpose();
	//calculate hidden derivative
	Eigen::MatrixXd hidden_derivative = _h->unaryExpr([=](double x)->double {
		return _ha->apply_derivative(x);
	});
	calc_hidden_delta(hidden_derivative, *_w1, out_delta, &hid_delta);
	//x->conservativeResize(x->rows() - 1, Eigen::NoChange);
	Eigen::MatrixXd hidden_gradient = hid_delta*_x->transpose();
	update_weights(alpha, hidden_gradient, output_gradient);
	//update bias using the sum of output deltas
	_b0 -= alpha*hid_delta.sum();
	_b1 -= alpha*out_delta.sum();
}

double neural_network::logistic_function(double x)
{
	return 1 / (1 + exp(-1 * x));
}

double neural_network::logistic_derivative(double x)
{
	return x*(1 - x);
}

void neural_network::softmax_function(Eigen::MatrixXd * x)
{
	Eigen::MatrixXd temp = x->unaryExpr([](double x)->double { return exp(x); });
	double sum = temp.sum();
	*x = temp.unaryExpr([=](double x) -> double { return x / sum; });
}


double neural_network::cross_entropy_loss(Eigen::MatrixXd & y, Eigen::MatrixXd & t)
{
	Eigen::MatrixXd tmp = y.unaryExpr([](double x)->double { return log(x); });
	return (-1)*(t.transpose()*tmp)(0, 0);
}

double neural_network::force_positive(double x)
{
	if (x < 0)
	{
		return (-1 * x);
	}
	return x;
}





