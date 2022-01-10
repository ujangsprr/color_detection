// g++ main.cpp -o main `pkg-config --libs --cflags opencv` && ./main

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <time.h>

#define H 		0
#define S		1
#define V		2
#define E		0
#define D		1
#define RECT	0
#define CROSS	1
#define ELIPSE	2

using namespace cv;
using namespace std;

int mode=0;

// Set Blue Color {H,S,V}
int YMin[3]={104,130,72},YMax[3]={130,242,170},YED[2]={1,2};

// Set Red Color {H,S,V}
int RMin[3]={149,113,51},RMax[3]={255,255,255},RED[2]={1,2};

// Set Green Color {H,S,V}
int GMin[3]={63,132,40},GMax[3]={93,240,118},GED[2]={0,1};

Mat erosion_src, dilation_src, erosion_dst, dilation_dst;
Mat imPart;
Mat frame;
Mat framedst;

int XB, YB;
int mapSdt;
int foundContours;
int numContours;
int indexterbesar;
float radius;
double terbesar;
double result0;
int detect;
float dt;
int count, countt;
int cekpix;
float PixlDis;
float RealHeight;
float RealWidth;

Point2f center;
Rect rect;


float PixtoReal(float distance)
{
	int disReal;
	
	disReal = 0+0.05*distance;

	return disReal;
}


Mat GetThresImage(Mat img, int *Min, int *Max){
	Mat imHSV, thres;
	cvtColor(img, imHSV, COLOR_BGR2HSV);

	inRange(imHSV, Scalar(Min[H],Min[S],Min[V]), Scalar(Max[H],Max[S],Max[V]), thres);

	return thres;
}

Mat Erosion(Mat img, int *ED){
	erosion_src = img;
	int erosion_type;
  	if(mode==RECT) erosion_type = MORPH_RECT;
  	else if(mode==CROSS) erosion_type = MORPH_CROSS;
  	else if(mode==ELIPSE) erosion_type = MORPH_ELLIPSE;

  	Mat element = getStructuringElement(erosion_type, Size(2*ED[E] + 1, 2*ED[E]+1 ), Point(ED[E], ED[E]) );
  	erode(erosion_src, erosion_dst, element);
	return erosion_dst;
}

Mat Dilation(Mat img, int *ED){
	dilation_src = img;

	int dilation_type;
  	if(mode==RECT) dilation_type = MORPH_RECT;
  	else if(mode==CROSS) dilation_type = MORPH_CROSS;
  	else if(mode==ELIPSE) dilation_type = MORPH_ELLIPSE;

  	Mat element = getStructuringElement( dilation_type, Size( 2*ED[D] + 1, 2*ED[D]+1 ), Point( ED[D], ED[D] ) );
  	dilate(dilation_src, dilation_dst, element );
    
	return dilation_dst;
}

void setLabel(cv::Mat& im, const std::string label, std::vector<cv::Point>& contour)
{
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    float scale = 0.4;
    int thickness = 1.5;
    int baseline = 0;
    cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    cv::Rect r = cv::boundingRect(contour);
    cv::Point pt(r.x + ((r.width - text.width) / 2), r.y + ((r.height + text.height) / 2));
    cv::putText(im, label, pt, fontface, scale, Scalar(255,255,0), thickness, 8);
}

void Detection(Mat &dst, Mat &frameTresh, String color)
{
	Mat frameDraw = Mat::zeros(frameTresh.size(), CV_8UC3);

    int numContours  =0;
	int foundContours=0;
	int indexterbesar=0;
	int terbesar=0;
	int result0 =0;
	rect= {0,0,0,0};

	vector<Vec4i> hierarchy;
	vector<vector<Point> > contours0;
	vector<vector<Point> > contours1;
	vector<Point> contours_poly;
	vector<Point> tampung;

    findContours(frameTresh, contours0, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0,0));

    vector<vector<Point> >hull(contours0.size());

	for(size_t i=0; i<contours0.size(); i++)
	{
		approxPolyDP(Mat(contours0[i]), contours_poly, 3, true);
		result0 = fabs(contourArea(contours0[i], false));
		rect = boundingRect(contours0[i]);
		minEnclosingCircle((Mat)contours_poly, center, radius);
		
		if(result0>terbesar && result0>20)
		{	
			terbesar = result0;
			indexterbesar = i;
	
			if(rect.width != frameTresh.cols)
			{
				numContours = indexterbesar; 
				foundContours = 1;		
			}
		}

		if(result0>terbesar && result0>20)
		{	
			terbesar = result0;
			indexterbesar = i;
	
			if(rect.width != frameTresh.cols)
			{
				numContours = indexterbesar; 
				foundContours = 1;
					
			}
		}
	}

	if(foundContours)
	{
		convexHull(contours0[numContours], hull[numContours]);
		drawContours(frameDraw, hull, (int)numContours, Scalar(255,255,255), CV_FILLED);
		// fillPoly(frameTresh,hull,Scalar(255,255,255));

		std::vector<cv::Point> approx;

		cv::approxPolyDP(cv::Mat(hull[numContours]), approx, cv::arcLength(cv::Mat(hull[numContours]), true)*0.02, true);
		int vtc = approx.size();
		
		rect = boundingRect(hull[numContours]);

		XB = rect.x + (rect.width/2);
		YB = rect.y + (rect.height/2);

		int width = ((rect.width/2)+(rect.height/2))/5;

		PixlDis = sqrt(pow((rect.x - rect.x),2)+pow((rect.y - (rect.y + rect.height)),2));
		
		RealWidth = PixtoReal(rect.width);
		RealHeight = PixtoReal(PixlDis);

		// cout << "Distance : " << PixlDis << " pixel" << endl;
		// cout << "Width  : " << RealWidth << " cm" << endl;
		// cout << "Height : " << RealHeight << " cm" << endl;
		// cout << "Titik  : " << vtc << endl << endl;

		Mat imCrop = frameDraw(rect);
		Mat framePot = Mat::zeros(imCrop.size(), CV_8UC3);

		bitwise_not(imCrop, framePot);
		
		imshow("Pot Frame", imCrop);
		imshow("Not Pot Frame", framePot);

		circle(dst, cvPoint(rect.x, rect.y), 8, Scalar(0, 255, 255), CV_FILLED, LINE_8);
		circle(dst, cvPoint(rect.x, rect.y + rect.height), 8, Scalar(0, 255, 255), CV_FILLED, LINE_8);
		circle(dst, cvPoint(rect.x + rect.width, rect.y), 8, Scalar(0, 255, 255), CV_FILLED, LINE_8);

        rectangle(dst, rect,  Scalar(0, 255, 255), 2, 8, 0);

		if(color == "Biru")
			putText(dst, format("Pot Biru %.0fx%.0f cm", RealHeight, RealWidth) , cvPoint(rect.x + 5, rect.y - 10), FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(0, 0, 255), 1, CV_AA);	
		else if(color == "Merah")
			putText(dst, format("Pot Merah %.0fx%.0f cm", RealHeight, RealWidth) , cvPoint(rect.x + 5, rect.y - 10), FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(0, 0, 255), 1, CV_AA);	
		else if(color == "Hijau")
			putText(dst, format("Pot Hijau %.0fx%.0f cm", RealHeight, RealWidth) , cvPoint(rect.x + 5, rect.y - 10), FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(0, 0, 255), 1, CV_AA);	
	}
}

void BlueDetect(Mat &frame, Mat &framedst)
{
    Mat thresh = GetThresImage(frame, YMin, YMax);
    Mat erosion = Erosion(thresh, YED);
    Mat dilation = Dilation(erosion, YED);

    Detection(framedst, dilation, "Biru");
    imshow("Yellow Treshold", dilation);
}

void RedDetect(Mat &frame, Mat &framedst)
{
    Mat thresh = GetThresImage(frame, RMin, RMax);
    Mat erosion = Erosion(thresh, RED);
    Mat dilation = Dilation(erosion, RED);

    Detection(framedst, dilation, "Merah");
    imshow("Red Treshold", dilation);
}

void GreenDetect(Mat &frame, Mat &framedst)
{
    Mat thresh = GetThresImage(frame, GMin, GMax);
    Mat erosion = Erosion(thresh, GED);
    Mat dilation = Dilation(erosion, GED);

    Detection(framedst, dilation, "Hijau");
    imshow("Green Treshold", dilation);
}

int main(int argc,char**argv)
{
    VideoCapture cap(2);

    // Mat image = imread("test.jpeg");

    while(1)
    {
        cap >> frame;

		resize(frame, frame, Size(720, 480), INTER_LINEAR);
        // frame = image;

        mode = ELIPSE;

        framedst = frame.clone();

        BlueDetect(frame, framedst);
        GreenDetect(frame, framedst);
        RedDetect(frame, framedst);

        imshow("Frame Input", frame);
        imshow("Frame Output", framedst);
        
        char key = cvWaitKey(33);
		if(key == 'q')
			break;
    }

    // cap.release();
	frame.release();
	cvDestroyAllWindows();
}