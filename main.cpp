// g++ main.cpp -o main `pkg-config --libs --cflags opencv` && ./main

// Pendefinisian library opencv dan c++
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <time.h>

// Pendefinisian variabel
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

// Nilai range HSV untuk warna biru + nilai erosi dilasi
int BMin[3]={104,130,72},BMax[3]={130,242,170},BED[2]={1,2};

// Nilai range HSV untuk warna merah + nilai erosi dilasi
int RMin[3]={149,113,51},RMax[3]={255,255,255},RED[2]={1,2};

// Nilai range HSV untuk warna merah + nilai erosi dilasi
int GMin[3]={63,132,40},GMax[3]={93,240,118},GED[2]={0,1};

// Pendefinisian matriks untuk menyimpan gambar dalam frame
Mat erosion_src, dilation_src, erosion_dst, dilation_dst;
Mat imPart;
Mat frame;
Mat framedst;

// Pendefinisian variabel global
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

// Fungsi rumus pixel ke real menggunakan regresi linier
float PixtoReal(float distance)
{
	int disReal;
	
	// rumus regresi linier (y=A+Bx), A = 0, B = 0.05, x = input pixel, y = output real
	disReal = 0+0.05*distance;

	return disReal;
}

// Fungsi thresholding gambar
Mat GetThresImage(Mat img, int *Min, int *Max){
	Mat imHSV, thres;

	// Konversi gambar dari RGB ke HSV
	cvtColor(img, imHSV, COLOR_BGR2HSV);

	// Threshold image dengan range warna HSV nilai minimal dan maksimal
	inRange(imHSV, Scalar(Min[H],Min[S],Min[V]), Scalar(Max[H],Max[S],Max[V]), thres);

	return thres;
}

// Fungsi erosi untuk menghapus kontur kecil pada frame threshold
Mat Erosion(Mat img, int *ED){
	erosion_src = img;
	int erosion_type;

	// Pendefinisian mode yang digunakan dalam erosi
  	if(mode==RECT) erosion_type = MORPH_RECT;
  	else if(mode==CROSS) erosion_type = MORPH_CROSS;
  	else if(mode==ELIPSE) erosion_type = MORPH_ELLIPSE;

	// Mengatur ukuran erosi yang digunakan berdasarkan input ED[E]
  	Mat element = getStructuringElement(erosion_type, Size(2*ED[E] + 1, 2*ED[E]+1 ), Point(ED[E], ED[E]) );
	// Menjalankan erosi dengan fungsi erode pada opencv
  	erode(erosion_src, erosion_dst, element);

	return erosion_dst;
}

// Fungsi dilasi untuk memperbesar kontur pada frame threshold
Mat Dilation(Mat img, int *ED){
	dilation_src = img;
	int dilation_type;

	// Pendefinisian mode yang digunakan dalam dilasi
  	if(mode==RECT) dilation_type = MORPH_RECT;
  	else if(mode==CROSS) dilation_type = MORPH_CROSS;
  	else if(mode==ELIPSE) dilation_type = MORPH_ELLIPSE;

	// Mengatur ukuran dilasi yang digunakan berdasarkan input ED[D]
  	Mat element = getStructuringElement( dilation_type, Size( 2*ED[D] + 1, 2*ED[D]+1 ), Point( ED[D], ED[D] ) );
	// Menjalankan dilate dengan fungsi erode pada opencv
  	dilate(dilation_src, dilation_dst, element );
    
	return dilation_dst;
}

// Fungsi deteksi objek berdasarkan kontur hasil thresholding + erosi + dilasi
void Detection(Mat &dst, Mat &frameTresh, String color)
{
	// Variabel matriks frameDraw sama dengan ukuran frameThresh dengan 3 channel warna
	Mat frameDraw = Mat::zeros(frameTresh.size(), CV_8UC3);

	// Varibel internal dalam fungsi
    int numContours  =0;
	int foundContours=0;
	int indexterbesar=0;
	int terbesar=0;
	int result0 =0;
	rect= {0,0,0,0};

	// Variabel vector untuk menyimpan kontur
	vector<Vec4i> hierarchy;
	vector<vector<Point> > contours0;
	vector<vector<Point> > contours1;
	vector<Point> contours_poly;
	vector<Point> tampung;

	// Mencari kontur yang ada pada frameThresh dan disimpan pada vector contours0
    findContours(frameTresh, contours0, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0,0));

	// Deklarasi variabel vector hull
    vector<vector<Point> >hull(contours0.size());

	// Looping berdasarkan banyaknya kontur
	for(size_t i=0; i<contours0.size(); i++)
	{
		// Mencari titik pada kontur
		approxPolyDP(Mat(contours0[i]), contours_poly, 3, true);
		// Menghitung besar kontur
		result0 = fabs(contourArea(contours0[i], false));
		// Membuat kotak pada kontur untuk mencari ukuran kontur
		rect = boundingRect(contours0[i]);
		// Membuat lingkaran pada kontur hasil approxPolyDP
		minEnclosingCircle((Mat)contours_poly, center, radius);
		
		// Jika kontur result0 lebih besar dari kontur terbesar dan nilai result0 leih dari 20
		if(result0>terbesar && result0>20)
		{	
			// result0 disimpan menjadi kontur terbesar
			terbesar = result0;
			// index terbesarnya disimpan berdasarkan nilai i dalam looping
			indexterbesar = i;
	
			// jika nilai lebar kontur tidak sama dengan lebar frame
			if(rect.width != frameTresh.cols)
			{
				// number contour adalah index terbesar
				numContours = indexterbesar; 
				// foundcontour = true
				foundContours = 1;		
			}
		}
	}

	//  jika nilai foundcontour bernilai true
	if(foundContours)
	{
		// kontur terbesar di convex hull
		convexHull(contours0[numContours], hull[numContours]);
		// menggambar kontur baru hasil convex hull
		drawContours(frameDraw, hull, (int)numContours, Scalar(255,255,255), CV_FILLED);

		// Deklarasi variabel vector approx
		std::vector<cv::Point> approx;

		// mencari jumlah titik pada kontur baru yang disimpan pada approx
		cv::approxPolyDP(cv::Mat(hull[numContours]), approx, cv::arcLength(cv::Mat(hull[numContours]), true)*0.02, true);
		// variabel vtc diisi dengan ukuran dari approx
		int vtc = approx.size();
		
		// bounding rect kontur untuk mencari ukuran kontur yang disimpan dalam rect
		rect = boundingRect(hull[numContours]);

		// mencari titik tengah (x,y) objek dengan nilai x,y kontur ditambah lebar,tinggi dibagi 2
		XB = rect.x + (rect.width/2);
		YB = rect.y + (rect.height/2);

		// Mencari nilai pixel tinggi dari objek dengan rumus jarak antara 2 titik
		PixlDis = sqrt(pow((rect.x - rect.x),2)+pow((rect.y - (rect.y + rect.height)),2));
		
		// konversi jarak pixel ke jarak real tinggi dan lebar
		RealWidth = PixtoReal(rect.width);
		RealHeight = PixtoReal(PixlDis);

		// cout << "Distance : " << PixlDis << " pixel" << endl;
		// cout << "Width  : " << RealWidth << " cm" << endl;
		// cout << "Height : " << RealHeight << " cm" << endl;
		// cout << "Titik  : " << vtc << endl << endl;

		// membuat frame baru dari nilai rect
		Mat imCrop = frameDraw(rect);
		// deklarasi matriks baru
		Mat framePot = Mat::zeros(imCrop.size(), CV_8UC3);

		// nilai frame imcrop dibalek dan disimpan di framepot
		bitwise_not(imCrop, framePot);
		
		// menampilkan frame ke layar
		imshow("Pot Frame", imCrop);
		imshow("Not Pot Frame", framePot);

		// membuat lingkaran pada titik-titik object rectangle (pojok kiri atas, pojok kiri bawah, pojok kanan atas)
		circle(dst, cvPoint(rect.x, rect.y), 8, Scalar(0, 255, 255), CV_FILLED, LINE_8);
		circle(dst, cvPoint(rect.x, rect.y + rect.height), 8, Scalar(0, 255, 255), CV_FILLED, LINE_8);
		circle(dst, cvPoint(rect.x + rect.width, rect.y), 8, Scalar(0, 255, 255), CV_FILLED, LINE_8);

		// membuat rectangle pada objek
        rectangle(dst, rect,  Scalar(0, 255, 255), 2, 8, 0);

		// menampilkan teks berdasarkan klasifikasi warna objek
		if(color == "Biru")
			putText(dst, format("Pot Biru %.0fx%.0f cm", RealHeight, RealWidth) , cvPoint(rect.x + 5, rect.y - 10), FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(0, 0, 255), 1, CV_AA);	
		else if(color == "Merah")
			putText(dst, format("Pot Merah %.0fx%.0f cm", RealHeight, RealWidth) , cvPoint(rect.x + 5, rect.y - 10), FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(0, 0, 255), 1, CV_AA);	
		else if(color == "Hijau")
			putText(dst, format("Pot Hijau %.0fx%.0f cm", RealHeight, RealWidth) , cvPoint(rect.x + 5, rect.y - 10), FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(0, 0, 255), 1, CV_AA);	
	}
}

// fungsi mendeteksi objek biru
void BlueDetect(Mat &frame, Mat &framedst)
{
	// Memanggil fungsi Threshold image dan disimpan ke matriks thresh
    Mat thresh = GetThresImage(frame, BMin, BMax);
	// Memanggil fungsi erosi dan disimpan ke matriks erosion
    Mat erosion = Erosion(thresh, BED);
	// Memanggil fungsi dilasai dan disimpan ke matriks dilation
    Mat dilation = Dilation(erosion, BED);

	// menjalankan fungsi deteksi dengan input frame dilation dan input parameter warna
    Detection(framedst, dilation, "Biru");
	// menampilkan frame ke layar
    imshow("Yellow Treshold", dilation);
}

// fungsi mendeteksi objek merah
void RedDetect(Mat &frame, Mat &framedst)
{
	// Memanggil fungsi Threshold image dan disimpan ke matriks thresh
    Mat thresh = GetThresImage(frame, RMin, RMax);
	// Memanggil fungsi erosi dan disimpan ke matriks erosion
    Mat erosion = Erosion(thresh, RED);
	// Memanggil fungsi dilasai dan disimpan ke matriks dilation
    Mat dilation = Dilation(erosion, RED);

	// menjalankan fungsi deteksi dengan input frame dilation dan input parameter warna
    Detection(framedst, dilation, "Merah");
	// menampilkan frame ke layar
    imshow("Red Treshold", dilation);
}

// fungsi mendeteksi objek hijau
void GreenDetect(Mat &frame, Mat &framedst)
{
	// Memanggil fungsi Threshold image dan disimpan ke matriks thresh
    Mat thresh = GetThresImage(frame, GMin, GMax);
	// Memanggil fungsi erosi dan disimpan ke matriks erosion
    Mat erosion = Erosion(thresh, GED);
	// Memanggil fungsi dilasai dan disimpan ke matriks dilation
    Mat dilation = Dilation(erosion, GED);

	// menjalankan fungsi deteksi dengan input frame dilation dan input parameter warna
    Detection(framedst, dilation, "Hijau");
	// menampilkan frame ke layar
    imshow("Green Treshold", dilation);
}

// main utama program dijalankan
int main(int argc,char**argv)
{
	// open camera berdasarkan index kamera
    VideoCapture cap(0);

	// jika kamera tidak terbuka maka akan menampilkan teks dibawah
	if(!cap.isOpened()){
		cout << "Can't open the camera" << endl;
	}

	// looping selama kamera terbuka
    while(cap.isOpened())
    {
		// cap dimasukkan ke frame
        cap >> frame;

		// mode erosi dilasi = ELIPSE
        mode = ELIPSE;

		// frame dst meng clonce dari frame
        framedst = frame.clone();

		// menggil fungsi deteksi biru dengan input frame
        BlueDetect(frame, framedst);
		// menggil fungsi deteksi hijau dengan input frame
        GreenDetect(frame, framedst);
		// menggil fungsi deteksi merah dengan input frame
        RedDetect(frame, framedst);

		// menampilkan frame ke layar
        imshow("Frame Input", frame);
        imshow("Frame Output", framedst);
        
		// membaca input pada keyboard
        char key = cvWaitKey(33);
		// jika input keyboard q maka program akan di break
		if(key == 'q')
			break;
    }

	// cap dan frame di release
    cap.release();
	frame.release();
	// destroy semua window
	cvDestroyAllWindows();
}