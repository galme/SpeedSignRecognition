#include <iostream>     // std::cout
#include <algorithm>    // std::shuffle
#include <array>        // std::array
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock

// ZA UÈENJE... lots of hardcoded stuff. :)

// preberi datoteko z nauèeno NN
Ptr<ml::ANN_MLP> loadANN()
{
	Ptr<ml::ANN_MLP> ann = ml::ANN_MLP::create();
	FileStorage fs;
	fs.open("NN_trained.yml", FileStorage::READ);
	ann->read(fs.getFirstTopLevelNode());

	return ann;
}

void trainANN(int nclasses, const Mat &train_data, const Mat &train_labels, const Mat &test_data, const Mat &test_labels, Mat &confusion)
{
	// setup the ann:
	int nfeatures = train_data.cols;
	Ptr<ml::ANN_MLP> ann = ml::ANN_MLP::create();

	Mat_<int> layers(3, 1);
	layers(0) = nfeatures;     // input
	layers(1) = nclasses * 8 * 2;  // hidden
	layers(2) = nclasses;      // output, 1 pin per class.
	ann->setLayerSizes(layers); 
	ann->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM, 0, 0);
	ann->setTermCriteria(TermCriteria(TermCriteria::Type::COUNT + TermCriteria::EPS, 42000, 0.00000000001));
	ann->setTrainMethod(ml::ANN_MLP::TrainingMethods::RPROP, 0.0001);
	
	// ann requires "one-hot" encoding of class labels:
	Mat train_classes = Mat::zeros(train_data.rows, nclasses, CV_32F);
	for (int i = 0; i<train_classes.rows; i++)
	{
		train_classes.at<float>(i, train_labels.at<int>(i)) = 1.f;
	}
	cerr << train_data.size() << " " << train_classes.size() << endl;

	ann->train(train_data, ml::ROW_SAMPLE, train_classes);

	// run tests on validation set:
	for (int i = 0; i<test_data.rows; i++) {
		Mat result;
		int pred = ann->predict(test_data.row(i), result);

		int truth = test_labels.at<int>(i);
		confusion.at<int>(pred, truth)++;
	}
	Mat correct = confusion.diag();
	float accuracy = sum(correct)[0] / sum(confusion)[0];
	cerr << "accuracy: " << accuracy << endl;
	cerr << "confusion:\n" << confusion << endl;

	ann->save("NN_trained.yml");
}

void learnAnn() // train wrapper
{
	const int nClasses = 10;
	string folders[nClasses] = { "Sample001", "Sample002","Sample003","Sample004","Sample005","Sample006","Sample007","Sample008","Sample009","Sample010" };
	int classSamples = 1016;

	Mat train_data(nClasses * classSamples, imgWidthAndLength * imgWidthAndLength, CV_32FC1);
	Mat labels(nClasses * classSamples, 1, CV_32S);

	for (int i = 1; i <= nClasses; i++)
	{
		Mat img;
		for (int j = 1; j <= classSamples; j++)
		{
			stringstream ss;
			ss << folders[i - 1] << "/img0";
			ss << ((i < 10) ? "0" : "");
			ss << i << "-";

			if (j < 10)
				ss << "0000";
			else if (j < 100)
				ss << "000";
			else if (j < 1000)
				ss << "00";
			else if (j < 10000)
				ss << "0";

			ss << j << ".png";
			
			img = imread(ss.str());
			cvtColor(img, img, cv::COLOR_BGR2GRAY);

			threshold(img, img, 0, 255, cv::THRESH_OTSU);
			int minx = 1000;
			int miny = 1000;
			int maxy = -1;
			int maxx = -1;

			for (int r = 0; r != img.rows; r++)
			{
				for (int c = 0; c != img.cols; c++)
				{
					float val = (img.at<uchar>(r, c) > 20) ? 0.0 : 1.0;
					if (val != 0.0)
					{
						if (r < miny)
							miny = r;
						if (r > maxy)
							maxy = r;
						if (c < minx)
							minx = c;
						if (c > maxx)
							maxx = c;
					}
				}
			}

			Rect ROI(minx, miny, maxx - minx, maxy - miny);
			img = img(ROI);

			cv::resize(img, img, Size(imgWidthAndLength, imgWidthAndLength));

			for (int r = 0; r != img.rows; r++)
			{
				for (int c = 0; c != img.cols; c++)
				{
					float val = (img.at<uchar>(r, c) > 20) ? 0.0 : 1.0;
					train_data.at<float>((i - 1) * classSamples + j - 1, r * img.rows + c) = val;
				} 
			}
			labels.row((i - 1) * classSamples + j - 1) = i - 1;
		}
	}

	int* arr = new int[nClasses * classSamples];
	for (int i = 0; i != nClasses * classSamples; i++)
		arr[i] = i;

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

	shuffle(arr, arr + nClasses * classSamples, std::default_random_engine(seed));

	Mat train_data_shuffled = train_data.clone();
	Mat lables_shuffled = labels.clone();
	for (int i = 0; i != nClasses * classSamples; i++)
	{
		train_data.row(i).copyTo(train_data_shuffled.row(arr[i]));
		labels.row(i).copyTo(lables_shuffled.row(arr[i]));
	}

	Mat confusion(nClasses, nClasses, CV_32S, Scalar(0)); // will hold our test results

	trainANN(nClasses, train_data, labels, train_data, labels, confusion);
}