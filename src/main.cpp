#include "svm.h"
#include "LoadData.h"
#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main()
{
	struct svm_parameter param;		// set by parse_command_line
	struct svm_problem prob;		// set by read_problem
	struct svm_problem probtest;
	struct svm_model *model;
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
	char *str = "C:\\Users\\zh\\Documents\\Visual Studio 2015\\Projects\\testSVM\\trainData.txt";
	char *strtest = "C:\\Users\\zh\\Documents\\Visual Studio 2015\\Projects\\testSVM\\testData.txt";
	prob = read_problem(str, param);
	probtest = read_problem(strtest, param);

	const char* check_param_err;
	check_param_err = svm_check_parameter(&prob, &param);
	if (check_param_err)
	{
		fprintf(stderr, "ERROR: %s\n", check_param_err);
		return -1;
	}

	model = svm_train(&prob, &param);

	double predictLabel[10];

	//double ans;
	//ans = svm_predict_values(model, probtest.x[90], predictLabel);

	//for (int i = 0;i != 10;++i) cout << predictLabel[i] << " ";
	//cout << endl;
	//cout << "ans is : " << ans << endl;

	int num = probtest.l;
	double index;
	double errorRate = 0;
	vector<double> label;
	for (int i = 0;i != num;++i)
	{
		index = svm_predict(model, probtest.x[i]);
		//cout << index << endl;
		label.push_back(index);
	}
	//double error = 0;
	for (int i = 0;i != num;++i)
	{
		if (label[i] != probtest.y[i])
		{
			errorRate += 1;
			cout << "Wrongly predicted data index is : " << i + 1 << ", truth label is: " << probtest.y[i] << ", predicted label is : " << label[i] << endl;
		}
	}
	cout << "Error Rate is : " << errorRate / probtest.l << endl;
	return 0;
}