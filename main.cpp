#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	Mat img = imread("crosswalk.jpg");
	imshow("loaded", img);
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);
	threshold(gray, gray, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	
	imshow("grayImg", gray);
	imwrite("grayimg.jpg", gray);
	waitKey(0);

	vector<vector<Point>> contours, tmpContours;
	vector<Vec4i> hierarchy;
	vector<vector<Point>> approx_list;
	findContours(gray, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_TC89_L1);
	for (int i = 0; i < contours.size(); i++) {
		
		double a = contourArea(contours[i], false);
		if (a > 1000) {
			vector<Point> approx;
			approxPolyDP(Mat(contours[i]), approx, 0.003 * arcLength(contours[i], true), true);
			approx_list.push_back(approx);
		}
	}
	//cout << approx_list.size() << endl;
	//for (int i = 0; i < approx_list.size(); i++) {
	//	drawContours(img, approx_list, i, Scalar(255, 255, 0, 100), -1, CV_AA, hierarchy, 0);
	//}
	//imshow("contours", img);
	//imwrite("contours.jpg", img);
	//waitKey(0);

	
	Mat fg = Mat::zeros(img.size(), img.type());
	for (int i = 0; i < approx_list.size(); i++) {
		drawContours(fg, approx_list, i, Scalar(255, 255, 0), -1, CV_AA, hierarchy, 0);
	}

	Mat fg_gray, fg_gray3;
	cvtColor(fg, fg_gray, CV_BGR2GRAY);
	
	threshold(fg_gray, fg_gray, 10, 255, CV_THRESH_BINARY);
	cvtColor(fg_gray, fg_gray3, CV_GRAY2BGR);
	Mat bg = img.mul(fg_gray3, 1.0/255.0);

	Mat overlay;
	addWeighted(fg, 0.5, bg, 0.5, 0, overlay);
	imwrite("overlay.jpg", overlay);

	Mat mask;
	bitwise_not(fg_gray3, mask);
	Mat masked = img.mul(mask, 1.0 / 255.0);
	Mat output;
	add(overlay, masked, output);
	imwrite("output.jpg", output);

}