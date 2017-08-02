#pragma once
using namespace std;
using namespace cv;

class ExtraBoardDetector
{
private:
	// za obe metodi
	const double maxSurfaceRatio = 0.3;

	// za normal-canny
	Rect searchArea;
	Mat edgeImg;
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	int ellipseHeight;

	// za dilated-canny
	Rect searchAreaDilated;
	Mat edgeImgDilated;
	vector< vector<Point> > contoursDilated;
	vector<Vec4i> hierarchyDilated;


	// dobi najnižjo (max Y) toèko v vektorju
	int getMaxYFromPoints(const vector<Point> const &pts)
	{
		int y = 0;
		for (int i = 0; i != pts.size(); i++)
		{
			if (pts[i].y > y)
				y = pts[i].y;
		}

		return y;
	}

public:

	ExtraBoardDetector(vector<Point> ellipse, const Mat const &edgeImage) : edgeImg(edgeImage)
	{
		// inicializiraj iskalno obmoèje na elipso
		searchArea = boundingRect(ellipse);
		ellipseHeight = searchArea.height;

		// premakni levo in poveèaj širino (da po širini zaobjamemo znak)
		searchArea.x -= searchArea.width / 3;
		searchArea.width += (int)(0.67 * searchArea.width);

		// trim x
		if (searchArea.x < 0)
			searchArea.x = 0;
		if (searchArea.width + searchArea.x > edgeImg.size().width)
			searchArea.width = edgeImg.size().width - searchArea.x;
		
		// pomakni dol (da preskoèimo speed-limit znak) in poveèaj višino (na cca 3 znake)
		searchArea.y += searchArea.height;
		searchArea.height *= 3.14;

		// trim y
		if (searchArea.y > edgeImg.size().height)
			searchArea.y = edgeImg.size().height;
		if (searchArea.height + searchArea.y > edgeImg.size().height)
			searchArea.height = edgeImg.size().height - searchArea.y;

		edgeImg = edgeImg(searchArea); // cropaj sliko na ROI

		// raširi/ojaèi canny robe ... ta primer kasneje obravnavamo posebej
		searchAreaDilated = searchArea;
		cv::morphologyEx(edgeImg, edgeImgDilated, cv::MORPH_DILATE, Mat::ones(Size(3, 3), CV_8U));

		// najdi konture v obeh podslikah
		findContours(edgeImg, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE); // najdi konture
		findContours(edgeImgDilated, contoursDilated, hierarchyDilated, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE); // najdi konture
	}

	// najdi dopolnilne znake s pomoèjo navadnih cannyevih robov
	void getBoundingBox(Mat const &rgb, Mat &result, bool dilatedEntryPoint = false)
	{
		int maxy = 0; // tu bo najveèji offset od konca znaka za omejitev hitrosti
		for (int i = 0; i != contours.size(); i++)
		{
			
			if (!dilatedEntryPoint)
			{
				// kontura je luknja v drugi ?
				if (hierarchy[i][Hierarchy::Parent] != -1)
					continue;
			}
			else
			{
				// zunanja kontura ?
				if (hierarchy[i][Hierarchy::Parent] == -1)
					continue;
			}

			// konture aproksimiraj s poligoni
			double epsilon = 10.0;
			vector<Point> polygon;
			approxPolyDP(contours[i], polygon, epsilon, true);

			// smo našli štirikotnik ?
			if (polygon.size() == 4)
			{
				// izraèunaj plošèini konture in aproksimiranega poligona
				double contourArea = cv::contourArea(contours[i]);
				double polyRectArea = minAreaRect(contours[i]).size.area();

				// ni dovolj podobno pravokotniku (po plošèini) ?
				if (fabs(polyRectArea / contourArea - 1) > maxSurfaceRatio)
					continue;

				// dobi spodnjo toèko dopolnilnega znaka
				int polyMaxY = getMaxYFromPoints(polygon);
				if (polyMaxY > maxy)
					maxy = polyMaxY;

				//cout << "contArea: " << contourArea << " | polyRectArea: " << polyRectArea << endl;
			}
		}

		// popravi "bounding box" na podroèje zaznanih znakov
		int extraOffset = ellipseHeight * 0.2;
		searchArea.height = maxy + ellipseHeight + 3 * extraOffset;
		searchArea.y -= ellipseHeight + extraOffset;

		result = rgb(searchArea);
	}

	// najdi dopolnilne znake s pomoèjo ojaèanih (dilated) cannyevih robov ... ne klièi "getBoundingBox" po tem!
	void getBoundingBoxDilated(const Mat const &rgb, Mat &result)
	{
		searchArea = searchAreaDilated;
		edgeImg = edgeImgDilated;
		contours = contoursDilated;
		hierarchy = hierarchyDilated;

		getBoundingBox(rgb, result, true);
	}
};