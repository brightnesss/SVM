#pragma once
#include "svm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))


static char* readline(FILE *input);
void exit_input_error(int line_num);
svm_problem read_problem(const char *filename, svm_parameter &param);