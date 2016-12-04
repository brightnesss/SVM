#pragma once
#include "svm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <map>
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))


static char* readline(FILE *input);
void exit_input_error(int line_num);
svm_problem read_problem(const char *filename, svm_parameter &param);
svm_parameter setParameter();
const char* checkParameter(svm_problem*, svm_parameter*);
double svmPredicted(svm_model*, svm_problem &,std::map<unsigned int,double> &);