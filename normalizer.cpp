#include "stdafx.h"
#include "normalizer.h"
#include "iostream"

normalizer::normalizer(Eigen::MatrixXd * matrix, double lb, double ub)
{
	this->_lb = lb;
	this->_ub = ub;
	this->_matrix = matrix;
	_colmap = new std::map<int, column_t>();
}

void normalizer::normalize_matrix()
{
	for (int i = 0; i < _matrix->cols(); i++) {
		double colmax = std::numeric_limits<double>::min();
		double colmin = std::numeric_limits<double>::max();
		for (int j = 0; j < _matrix->rows(); j++) {
			if ((*_matrix)(j, i) > colmax) {
				colmax = (*_matrix)(j, i);
			}
			if ((*_matrix)(j, i) < colmin) {
				colmin = (*_matrix)(j, i);
			}
		}
		column_t col;
		col.column_min = colmin;
		col.column_max = colmax;
		_colmap->insert(std::pair<int, column_t>(i, col));
		for (int j = 0; j < _matrix->rows(); j++) {
			double value = _lb + (_ub - _lb) * ((*_matrix)(j, i) - colmin) / (colmax - colmin);
			(*_matrix)(j, i) = value;
		}
	}	
}

void normalizer::normalize_sample(Eigen::MatrixXd * sample)
{
	int i = 0;
	*sample = sample->unaryExpr([&](double x) -> double {
		std::map<int, column_t>::iterator iter;		
		double colmin = 0, colmax = 0;
		iter = _colmap->find(i++);
		if (iter != _colmap->end()) {
			column_t col = iter->second;
			colmin = col.column_min;
			colmax = col.column_max;
		}
		return _lb + (_ub - _lb) * ((x - colmin) / (colmax - colmin));
	});
}


normalizer::~normalizer()
{
	if (_colmap) {
		delete _colmap;
	}
}
