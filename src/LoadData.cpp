#include "LoadData.h"
#include "svm.h"

static int max_line_len;
static char *line = NULL;

static char* readline(FILE *input)
{
	int len;

	if (fgets(line, max_line_len, input) == NULL)
		return NULL;

	while (strrchr(line, '\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *)realloc(line, max_line_len);
		len = (int)strlen(line);
		if (fgets(line + len, max_line_len - len, input) == NULL)
			break;
	}
	return line;
}

void exit_input_error(int line_num)
{
	fprintf(stderr, "Wrong input format at line %d\n", line_num);
	exit(1);
}

svm_problem read_problem(const char *filename, svm_parameter &param)
{
	svm_problem prob;
	struct svm_node *x_space;
	int max_index, inst_max_index, i;
	size_t elements, j;
	FILE *fp = fopen(filename, "r");
	char *endptr;
	char *idx, *val, *label;

	if (fp == NULL)
	{
		fprintf(stderr, "can't open input file %s\n", filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;

	max_line_len = 1024;
	line = Malloc(char, max_line_len);
	while (readline(fp) != NULL)
	{
		char *p = strtok(line, " \t"); // label 用" "和"\t"问分隔符，分割line

									   // features
		while (1)
		{
			p = strtok(NULL, " \t");
			if (p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			++elements;
		}
		++elements;
		++prob.l;
	}
	rewind(fp);

	prob.y = Malloc(double, prob.l);
	prob.x = Malloc(struct svm_node *, prob.l);
	x_space = Malloc(struct svm_node, elements);

	max_index = 0;
	j = 0;
	for (i = 0;i<prob.l;i++)
	{
		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line, " \t\n");
		if (label == NULL) // empty line
			exit_input_error(i + 1);

		prob.y[i] = strtod(label, &endptr);
		if (endptr == label || *endptr != '\0')
			exit_input_error(i + 1);

		while (1)
		{
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if (val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int)strtol(idx, &endptr, 10);
			if (endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i + 1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val, &endptr);
			if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i + 1);

			++j;
		}

		if (inst_max_index > max_index)
			max_index = inst_max_index;
		x_space[j++].index = -1;
	}

	if (param.gamma == 0 && max_index > 0)
		param.gamma = 1.0 / max_index;

	if (param.kernel_type == PRECOMPUTED)
		for (i = 0;i<prob.l;i++)
		{
			if (prob.x[i][0].index != 0)
			{
				fprintf(stderr, "Wrong input format: first column must be 0:sample_serial_number\n");
				exit(1);
			}
			if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				fprintf(stderr, "Wrong input format: sample_serial_number out of range\n");
				exit(1);
			}
		}

	fclose(fp);
	return prob;
}

svm_parameter setParameter()
{
	struct svm_parameter param;
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 1 / 13;	// 1/num_features
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	return param;
}

const char* checkParameter(svm_problem *prob, svm_parameter *param)
{
	const char* check_param_err;
	check_param_err = svm_check_parameter(prob, param);
	if (check_param_err)
	{
		fprintf(stderr, "ERROR: %s\n", check_param_err);
	}
	return check_param_err;
}

double svmPredicted(svm_model *model, svm_problem &probtest, std::map<unsigned int, double> &badNum)
{
	double error = 0;
	unsigned int num = probtest.l;
	double index;
	for (unsigned int i = 0;i != num;++i)
	{
		index = svm_predict(model, probtest.x[i]);
		if (index != probtest.y[i])
		{
			error += 1;
			badNum.insert(std::make_pair(i + 1, index));
		}
	}
	return error / probtest.l;
}