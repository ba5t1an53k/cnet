#pragma once
#include <Eigen/Dense>
#include <fstream>
#include <vector>

#define COLUMN_DELIM ' '
#define IDX_VAL_DELIM ':'

class data_source
{
	std::string _data_path;
	Eigen::MatrixXd* _dcontainer;
	Eigen::MatrixXd* _tcontainer;
public:
	explicit data_source(std::string data_path);
	void get_data_matrix(Eigen::MatrixXd** data_matrix, Eigen::MatrixXd** target_matrix);
	~data_source();
};

