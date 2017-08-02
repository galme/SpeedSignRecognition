#pragma once
using namespace cv;
using namespace std;

class EllipseDetector
{
private:

	const vector< vector<Point> > contours;
	const vector<Vec4i> hierarchy; // [Next, Previous, First_Child, Parent]
	const double axisErrorThreshold = 0.20;
	const double areaErrorThreshold = 0.05;

public:
	vector< vector<Point> > ellipses;

	// constructors
	EllipseDetector(const vector< vector<Point> > const &contours, const vector<Vec4i> const &hierarchy) : contours(contours), hierarchy(hierarchy)
	{

	}

	void detectEllipses()
	{
		double a = 0;
		double b = 0;
		double estimatedArea = 0;
		double contourArea = 0;
		double ratio = 0;

		for (int i = 0; i != contours.size(); i++)
		{
			if (hierarchy[i][Hierarchy::Parent] == -1)
			{
				cv::Rect rect = boundingRect(contours[i]);
				
				// objekt ni premajhen ?
				if (rect.width > 30 && rect.height > 30)
				{
					RotatedRect fittedEllipse = fitEllipse(contours[i]);
					a = fittedEllipse.size.width / 2.0;
					b = fittedEllipse.size.height / 2.0;

					estimatedArea = M_PI * a * b;
					contourArea = fabs(cv::contourArea(contours[i]));

					ratio = estimatedArea / contourArea;

					// se plošèina ujema in sta si glavni osi podobni ?
					if (fabs(ratio - 1.0) <= areaErrorThreshold && fabs(a / b - 1.0) <= axisErrorThreshold)
					{
						cout << "Ellipse found ... Surface area ratio: " << ratio << " " << a << " " << b << endl;
						ellipses.push_back(contours[i]);
					}
				}
			}
		}
	}

	~EllipseDetector()
	{
	}
};

