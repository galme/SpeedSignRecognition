#pragma once
#include <algorithm>
#include "Number.h"

class NumberDetector
{
private:
	// neodvinse od metode
	const vector<Point> ellipse;
	int ellipseWidth, ellipseHeight;
	Ptr<ml::ANN_MLP> ann;

	// za normal-canny
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy; // [Next, Previous, First_Child, Parent]
	Mat numberMask;

	// za dilated-canny
	vector< vector<Point> > contoursDilated;
	vector<Vec4i> hierarchyDilated; // [Next, Previous, First_Child, Parent]
	Mat numberMaskDilated;


	// najde top-left toèko v konturi (@contour) in jo premakne malenkost dol in desno (odvisno od @offset)
	Point findSeed(const vector<Point> const &contour, int maxy)
	{
		Point bottomLeft;
		int mindistance = 1000;
		for (int i = 0; i != contour.size(); i++)
		{
			int calcuclatedDistance = contour[i].x + (maxy - contour[i].y);
			if (calcuclatedDistance < mindistance)
			{
				mindistance = calcuclatedDistance;
				bottomLeft = contour[i];
			}
		}

		return bottomLeft;
	}

	void moveSeed(Point &seed, const Mat const &mask)
	{
		for (int i = 0; i != 10; i++)
		{
			if (mask.at<uchar>(seed.y, seed.x) < 20)
				break;

			seed.x += 1;
			seed.y -= 1;
		}
	}

	bool validateResult(const vector<Number> const &numbers, int &speed)
	{
		bool ok = false;
		// imamo 1-3 števke?
		if (numbers.size() == 1)
		{
			speed = numbers[0].num;
			if (speed != 5)
			{
				throw exception("hitrost z eno stevko, ki ni 5");
			}
			ok = true;
		}
		else if (numbers.size() == 2)
		{
			speed = numbers[0].num * 10 + numbers[1].num;
			if (speed < 20 && speed != 15)
			{
				throw exception("hitrost z dvema stevka pod 20, ki pa ni 15");
			}
			else if (speed > 20 && speed % 10 != 0)
			{
				throw exception("hitrost z dvema stevka nad 20, pri kateri 2. stevka ni 0");
			}
			ok = true;
		}
		else if (numbers.size() == 3)
		{
			speed = numbers[0].num * 100 + numbers[1].num * 10 + numbers[2].num;

			if (numbers[2].num != 0)
			{
				throw exception("hitrost s tremi stevkami, pri kateri zadnja ni 0!");
			}
			if (speed > 130)
			{
				throw exception("hitrost nad 130... to ne bo prav :)");
			}
			ok = true;
		}
		else if (numbers.size() > 3)
		{
			string err = "weird sign (veè kot 3 števke) .... razpoznano: ";
			for (int i = 0; i != numbers.size(); i++)
				err += numbers[i].num;
			throw exception(err.c_str());
		}

		return ok;
	}

public:
	Mat transformMatrix; // transormacijska matrika za poravnavo (znaka)
	Mat transformMatrixDilated; // transormacijska matrika za poravnavo (znaka)
	int speed = -1; // razpoznana hitrost iz znaka

	NumberDetector(const Ptr<ml::ANN_MLP> const &ann, const vector<Point> const &detectedEllipse, const Mat const &edgeImg) : ann(ann), ellipse(detectedEllipse)
	{
		// ustvari masko, ki predstavlja zunanji del elipse
		Rect ellipseRect = boundingRect(ellipse);
		Mat contourMask = Mat::zeros(ellipseRect.size(), CV_8U);
		contours.push_back(ellipse);
		drawContours(contourMask, contours, 0, Scalar(255), cv::FILLED, 8, noArray(), 100000, Point(-ellipseRect.x, -ellipseRect.y));
		bitwise_not(contourMask, contourMask);

		// odstranimo del, ki je zunaj elipse
		numberMask = edgeImg.clone()(ellipseRect); // this clone thing...
		morphologyEx(contourMask, contourMask, cv::MORPH_DILATE, Mat::ones(Size(5, 5), CV_8U));
		numberMask -= contourMask;

		// ker vèasih canny zafrkne... še malo ojaèamo robe. Ta primer z ojaèanimi robi obravnavamo posebej (kot backup)
		morphologyEx(numberMask, numberMaskDilated, cv::MORPH_DILATE, Mat::ones(Size(3, 3), CV_8U));

		// najdi konture v sliki z navadnimi canny robi
		contours.clear();
		findContours(numberMask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

		// najdi konture v sliki z ojaèanimi (dilated) canny robi
		contoursDilated.clear();
		findContours(numberMaskDilated, contoursDilated, hierarchyDilated, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

		ellipseWidth = ellipseRect.width;
		ellipseHeight = ellipseRect.height;
	}

	// najde števke glede na navadne canny robe
	bool findNumbers(bool findDilatedNumbersEntryPoint = false)
	{
		if (!findDilatedNumbersEntryPoint)
			cout << "DETECTING NUMBERS (normal canny) " << endl;
		vector<Number> numbers;

		// najdi preslikavo, ki bo poravnala števke
		for (int i = 0; i != contours.size(); i++)
		{
			// nismo na najbolj zunanji konturi ?
			if (hierarchy[i][Hierarchy::Parent] != -1)
				continue; // @SHADY ... no continue ?

			// filtriranje po velikosti:
			Rect numberRect = boundingRect(contours[i]);
			if (numberRect.height < 20 || numberRect.height < ellipseHeight / 4 || numberRect.width * numberRect.height >(ellipseWidth * ellipseHeight) / 2)
				continue;

			RotatedRect rotatedNumberRect = minAreaRect(contours[i]);

			// je rotacija veè kot 45 stopinj ? ... potem se dogajajo èudne stvari --> ignore
			if (fabs(rotatedNumberRect.angle) > 45)
				rotatedNumberRect.angle = 0;

			// najdi transformacijo, ki bo poravnala znak, ter transformiraj (pod)sliko s števkami
			transformMatrix = getRotationMatrix2D(rotatedNumberRect.center, rotatedNumberRect.angle, 1);
			cv::warpAffine(numberMask, numberMask, transformMatrix, numberMask.size());

			break; // transformacijo smo ravnokar dobili ...
		}

		// najdi števke
		for (int i = 0; i != contours.size() && transformMatrix.rows != 0; i++)
		{
			Number numberData;

			// preskoèimo vse, razen najbolj zunanjih
			if (hierarchy[i][Hierarchy::Parent] != -1)
				continue;

			// transformiraj konturo
			transform(contours[i], contours[i], transformMatrix);

			// filtriranje po velikosti:
			Rect numberRect = boundingRect(contours[i]);
			if (numberRect.height < 20 || numberRect.height < ellipseHeight / 4 || numberRect.width * numberRect.height > (ellipseWidth * ellipseHeight) / 2)
				continue;

			
			Mat numberMask_COPY = numberMask.clone();

			// iskanje polnilne toèke (seed)
			Point seed = findSeed(contours[i], numberRect.height + numberRect.y); // najdemo toèko znotraj konture
			moveSeed(seed, numberMask_COPY);

			// èuden seed ?
			if (seed.x > numberMask_COPY.cols || seed.y > numberMask_COPY.rows)
				continue;

			// polnjenje konture števke
			floodFill(numberMask_COPY, seed, Scalar(255), 0, Scalar(20), Scalar(100), cv::FLOODFILL_FIXED_RANGE);

			// trim
			if (numberRect.x + numberRect.width > numberMask_COPY.cols)
				numberRect.width = numberMask_COPY.cols - numberRect.x;
			if (numberRect.y + numberRect.height > numberMask_COPY.rows)
				numberRect.height = numberMask_COPY.rows - numberRect.y;

			Mat number = numberMask_COPY(numberRect);
			numberData.boundingRect = numberRect;

			// posebej preverimo, èe je enka... ker jo resize pokvari za NN
			bool couldBeOne = false;
			if (numberData.boundingRect.width <= numberData.boundingRect.height * 0.3)
			{
				couldBeOne = true;
			}
			
			int res = -1;
			if (!couldBeOne)
			{
				// spremenimo velikost, da bo primerna za vhod v NN in ustvarjanje dejanskega vhodnega vzorca
				cv::resize(number, number, Size(imgWidthAndLength, imgWidthAndLength));
				Mat pattern(1, imgWidthAndLength * imgWidthAndLength, CV_32F);
				for (int r = 0; r != number.rows; r++)
				{
					for (int c = 0; c != number.cols; c++)
					{
						float val = (number.at<uchar>(r, c) > 20) ? 1.0 : 0.0;
						pattern.at<float>(0, r * number.rows + c) = val;
					}
				}

				// predikcija z NN in shranjevanje rezultata
				res = ann->predict(pattern);
			}
			else
			{
				res = 1;
			}
			
			numberData.num = res;
			numbers.push_back(numberData);
		}

		// sortiraj števke po X osi
		sort(numbers.begin(), numbers.end(), Number::numberDataComparator);

		bool ok = validateResult(numbers, speed);
		
		// uspešno razpoznali ?
		if (ok)
		{
			cout << "Detection successful, SPEED: " << speed << endl;
			return true;
		}

		return false;
	}

	// najde števke glede na ojaèane canny robe ... ne klièi "findNumbers" po tem!
	bool findDilatedNumbers()
	{
		cout << "DETECTING NUMBERS (dilated canny) " << endl;
		contours = contoursDilated;
		hierarchy = hierarchyDilated;
		numberMask = numberMaskDilated;

		return findNumbers(true);
	}
};