#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <time.h>
#include <fstream>

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

void trackBar_update();

int mode=0;
// int wMin[3]={0,0,0},wMax[3]={255,255,255},ED[3]={0,0,0};
int wMin[3]={0,115,157},wMax[3]={207,162,228},ED[3]={0,1,0};

Mat erosion_src, dilation_src, erosion_dst, dilation_dst;
Mat imPart;
Mat frame;

Mat GetThresImage(Mat img){
	Mat imHSV, thres;
	cvtColor(img, imHSV, COLOR_BGR2HSV);

	inRange(imHSV, Scalar(wMin[H],wMin[S],wMin[V]), Scalar(wMax[H],wMax[S],wMax[V]), thres);

	return thres;
}

Mat Erosion(Mat img){
	erosion_src = img;
	int erosion_type;
  	if(mode==RECT) erosion_type = MORPH_RECT;
  	else if(mode==CROSS) erosion_type = MORPH_CROSS;
  	else if(mode==ELIPSE) erosion_type = MORPH_ELLIPSE;

  	Mat element = getStructuringElement(erosion_type, Size(2*ED[E] + 1, 2*ED[E]+1 ), Point(ED[E], ED[E]) );
  	erode(erosion_src, erosion_dst, element);
	return erosion_dst;
}

Mat Dilation(Mat img){
	dilation_src = img;

	int dilation_type;
  	if(mode==RECT) dilation_type = MORPH_RECT;
  	else if(mode==CROSS) dilation_type = MORPH_CROSS;
  	else if(mode==ELIPSE) dilation_type = MORPH_ELLIPSE;

  	Mat element = getStructuringElement( dilation_type, Size( 2*ED[D] + 1, 2*ED[D]+1 ), Point( ED[D], ED[D] ) );
  	dilate(dilation_src, dilation_dst, element );
    
	return dilation_dst;
}

void trackBar(){
	cvCreateTrackbar("H/Y MIN", "Result", &wMin[H], 255, 0);
	cvCreateTrackbar("S/U MIN", "Result", &wMin[S], 255, 0);
	cvCreateTrackbar("V MIN", "Result", &wMin[V], 255, 0);
	cvCreateTrackbar("H/Y MAX", "Result", &wMax[H], 255, 0);
	cvCreateTrackbar("S/U MAX", "Result", &wMax[S], 255, 0);
	cvCreateTrackbar("V MAX", "Result", &wMax[V], 255, 0);
	cvCreateTrackbar("E", "Result", &ED[E], 100, 0);
	cvCreateTrackbar("D", "Result", &ED[D], 100, 0);
}

void trackBar_update(){
	cvSetTrackbarPos("H/Y MIN", "Result", wMin[H]);
	cvSetTrackbarPos("S/U MIN", "Result", wMin[S]);
	cvSetTrackbarPos("V MIN", "Result", wMin[V]);
	cvSetTrackbarPos("H/Y MAX", "Result", wMax[H]);
	cvSetTrackbarPos("S/U MAX", "Result", wMax[S]);
	cvSetTrackbarPos("V MAX", "Result", wMax[V]);
	cvSetTrackbarPos("E", "Result", ED[E]);
	cvSetTrackbarPos("D", "Result", ED[D]);
}

int main(int argc,char**argv)
{
	ofstream myfile;
	myfile.open("result.txt");
	
	int count = 0;

    VideoCapture cap(2);
    // Mat image = imread("image0.jpg");

    while(1)
    {
        cap >> frame;
        // frame = image;

        Mat thresh = GetThresImage(frame);
		Mat erosion = Erosion(thresh);
		Mat dilation = Dilation(erosion);

        cvNamedWindow("Result");
        trackBar();
        // trackBar_update();
        mode = CROSS;

        imshow("Frame", frame);
        imshow("Threshold", dilation);
        
		char key = cvWaitKey(33);
		if(key == 's')
		{
			char buff[50];

			printf("\n\nMin[3]={%d,%d,%d}\n", wMin[H], wMin[S], wMin[V]);
			printf("Max[3]={%d,%d,%d}\n", wMax[H], wMax[S], wMax[V]);
			printf("ED[2]={%d,%d}\n\n", ED[E], ED[D]);

			sprintf(buff, "Min[3]={%d,%d,%d}\nMax[3]={%d,%d,%d}\nED[2]={%d,%d}\n\n", wMin[H], wMin[S], wMin[V], wMax[H], wMax[S], wMax[V], ED[E], ED[D]);

			myfile << buff;
			
		}
		else if(key == 'q')
		{
			return 0;
		}
		else if(key == 'c')
		{
			char name[20];
			sprintf(name,"image%d.jpg", count);
			imwrite(name, frame);
			count++;
		}
    }

	myfile.close();
    // cap.release();
	frame.release();
	cvDestroyAllWindows();
}