#include "svm.h"
#include "LoadData.h"
#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main()
{
#include "svm.h"
#include "Data.h"
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
	param = setParameter();
	//char *str = "trainData.txt";
	//char *strtest = "testData.txt";
	string featureStr, testFeatureStr, labelStr, testLabelStr;
	featureStr = "train_feature.txt";
	labelStr = "train_label.txt";
	testFeatureStr = "test_feature.txt";
	testLabelStr = "test_label.txt";
	vector<vector<double>> trainFeature, trainLabel, testFeature, testLabel;
	load_feature(featureStr, trainFeature);
	load_feature(labelStr, trainLabel);

	vector< vector<double> > normalizeparam;
	normalizeparam = normalize_trainFeature(trainFeature);

	prob = init_svm_problem(trainFeature, trainLabel, param);

	load_feature(testFeatureStr, testFeature);
	load_feature(testLabelStr, testLabel);

	normalize_testFeature(testFeature, normalizeparam);
	probtest = init_svm_problem(testFeature, testLabel, param);


	//prob = read_problem(str, param);
	//probtest = read_problem(strtest, param);

	const char* check_param_err;
	check_param_err = checkParameter(&prob, &param);

	model = svm_train(&prob, &param);

	double errorRate;
	std::map<unsigned int, double> badNum;
	errorRate = svmPredicted(model, probtest, badNum);
	cout << "Error Rate is : " << errorRate << endl;
	map<unsigned int, double>::iterator iter = badNum.begin();
	for (;iter!=badNum.end();++iter)
	{
		cout << "Wrongly Predicted Data Index is : " << iter->first << ", " << "Wrongly Predicted Label is : " << iter->second << endl;
	}
}