#pragma once


#include "svm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <map>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

namespace mysvm
{
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))


	static char* readline(FILE *input);
	void exit_input_error(int line_num);
	svm_problem read_problem(const char *filename, svm_parameter &param);
	svm_parameter setParameter();
	const char* checkParameter(svm_problem*, svm_parameter*);
	double svmPredicted(svm_model*, svm_problem &, std::map<unsigned int, double> &);
	std::vector<double> norateSVMPredicted(svm_model*, svm_problem &);
	svm_problem init_svm_problem(std::vector<std::vector<double> > &feature, std::vector<double> &label, svm_parameter &param);
	int load_feature(std::string &filename, std::vector< std::vector<double> > &feature);
	int load_label(std::string &filename, std::vector<double> &label);
	std::vector< std::vector<double> > normalize_trainFeature(std::vector< std::vector<double> > &feature);
	void normalize_testFeature(std::vector<std::vector<double>>& feature, std::vector<std::vector<double>>& para);
}