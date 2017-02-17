#include "svm.h"
#include "LoadData.h"

#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main()
{
	struct mysvm::svm_parameter param;		// set by parse_command_line
	struct mysvm::svm_problem prob;		// set by read_problem
	struct mysvm::svm_problem probtest;
	struct mysvm::svm_model *model;
	param = mysvm::setParameter();
	//char *str = "trainData.txt";
	//char *strtest = "testData.txt";
	string featureStr, testFeatureStr, labelStr, testLabelStr;
	featureStr = "train_feature.txt";
	labelStr = "train_label.txt";
	testFeatureStr = "test_feature.txt";
	testLabelStr = "test_label.txt";
	vector<vector<double>> trainFeature, testFeature;
	vector<double> trainLabel, testLabel;
	mysvm::load_feature(featureStr, trainFeature);
	mysvm::load_label(labelStr, trainLabel);

	vector< vector<double> > normalizeparam;
	normalizeparam = mysvm::normalize_trainFeature(trainFeature);

	prob = mysvm::init_svm_problem(trainFeature, trainLabel, param);

	mysvm::load_feature(testFeatureStr, testFeature);
	mysvm::load_label(testLabelStr, testLabel);

	mysvm::normalize_testFeature(testFeature, normalizeparam);
	probtest = mysvm::init_svm_problem(testFeature, testLabel, param);


	//prob = read_problem(str, param);
	//probtest = read_problem(strtest, param);

	const char* check_param_err;
	check_param_err = mysvm::checkParameter(&prob, &param);

	model = mysvm::svm_train(&prob, &param);

	double errorRate;
	std::map<unsigned int, double> badNum;
	errorRate = mysvm::svmPredicted(model, probtest, badNum);
	cout << "Error Rate is : " << errorRate << endl;
	map<unsigned int, double>::iterator iter = badNum.begin();
	for (;iter!=badNum.end();++iter)
	{
		cout << "Wrongly Predicted Data Index is : " << iter->first << ", " << "Wrongly Predicted Label is : " << iter->second << endl;
	}
}