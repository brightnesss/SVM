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
	param=setParameter();
	char *str = "C:\\Users\\zh\\Documents\\Visual Studio 2015\\Projects\\testSVM\\trainData.txt";
	char *strtest = "C:\\Users\\zh\\Documents\\Visual Studio 2015\\Projects\\testSVM\\testData.txt";
	prob = read_problem(str, param);
	probtest = read_problem(strtest, param);

	const char* check_param_err;
	check_param_err=checkParameter(&prob,&param);

	model = svm_train(&prob, &param);

	double errorRate;
	std::map<unsigned int,double> badNum;
	errorRate=svmPredicted(model,probtest,badNum);
	cout << "Error Rate is : " << errorRate << endl;
	map<unsigned int, double>::iterator iter = badNum.begin();
	for (;iter!=badNum.end();++iter)
	{
		cout << "Wrongly Predicted Data Index is : " << iter->first << ", " << "Wrongly Predicted Label is : " << iter->second << endl;
	}
	return 0;
}