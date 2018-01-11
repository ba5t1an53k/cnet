#include "stdafx.h"
#include "data_source.h"
#include <iostream>

data_source::data_source(std::string data_path)
{
	this->_data_path = data_path;
}

int count_lines(const std::string &s, char delim) {
	std::stringstream ss(s);
	std::string item;
	int count = 0;
	while (std::getline(ss, item, delim)) {
		count++;
	}
	return count;
}

void get_idx_value(const std::string &s, char delim, int* idx, double* value) {
	std::stringstream ss(s);
	std::string item;
	int i = 0;
	while (std::getline(ss, item, delim)) {
		if (i == 0)
		{
			*idx = std::stoi(item);
			i++;
		}
		else 
		{
			*value = std::stod(item);
		}
	}
}

void get_row_vector(const std::string &s, char delim, Eigen::MatrixXd * row, int * label) {
	std::stringstream ss(s);
	std::string item;
	int i = 0;
	while (std::getline(ss, item, delim)) {
		if (i == 0)
		{
			*label = std::stoi(item);
			++i;
		}
		else
		{
			int idx = 0;
			double value = 0;
			get_idx_value(item,IDX_VAL_DELIM, &idx, &value);
			(*row)(0, idx - 1) = value;
		}
	}
}

void data_source::get_data_matrix(Eigen::MatrixXd ** data_matrix, Eigen::MatrixXd** target_matrix)
{
	if (_dcontainer)
	{
		data_matrix = &_dcontainer;
		target_matrix = &_tcontainer;
		return;
	}
	std::ifstream file(_data_path);
	int num_rows = std::count(std::istreambuf_iterator<char>(file),
		std::istreambuf_iterator<char>(), '\n') + 1;
	_tcontainer = new Eigen::MatrixXd(num_rows,1);
	file.clear();
	file.seekg(0, std::ios::beg);
	std::string line;
	int num_columns = 0;
	int row_counter = 0;
	while (std::getline(file, line)) 
	{
		if (_dcontainer == NULL)
		{
			num_columns = count_lines(line, COLUMN_DELIM) - 1;
			_dcontainer = new Eigen::MatrixXd(num_rows, num_columns);
		}
		Eigen::MatrixXd row(1, num_columns);
		int label = 0;
		get_row_vector(line, COLUMN_DELIM, &row, &label);
		(*_dcontainer).row(row_counter) = row;
		(*_tcontainer)(row_counter++) = label;
	}
	std::cout << *_dcontainer << std::endl;
	*data_matrix = _dcontainer;
	*target_matrix = _tcontainer;
}

data_source::~data_source()
{
	if (_dcontainer)
	{
		delete _dcontainer;
	}
	if (_tcontainer)
	{
		delete _tcontainer;
	}
}

