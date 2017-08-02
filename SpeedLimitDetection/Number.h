#pragma once
using namespace std;
using namespace cv;
class Number
{

public:
	int num;
	int probabilityVector[10];
	Rect boundingRect;

	int static numberDataComparator(Number &a, Number &b)
	{
		return a.boundingRect.x < b.boundingRect.x;
	}
};