#pragma once
#include <Eigen\Dense>
#include <map>

struct column_t {
	double column_max;
	double column_min;
};

class normalizer
{
private:
	Eigen::MatrixXd* _matrix;
	std::map<int, column_t>* _colmap;
	double _lb;
	double _ub;
public:
	explicit normalizer(Eigen::MatrixXd* matrix, double lb, double ub);
	void normalize_matrix();
	void normalize_sample(Eigen::MatrixXd* sample);
	~normalizer();
};

