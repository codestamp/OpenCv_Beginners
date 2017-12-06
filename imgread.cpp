#include "opencv2/core/utility.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda_types.hpp>


using namespace cv;


int main(int argc,char** argv) {

	Mat result,img,cannyres;
	img=imread(argv[1],IMREAD_COLOR);
	cuda::GpuMat src,dst;

	src.upload(img);

	cuda::bilateralFilter(src,dst,-1,50,7);
	dst.download(result);

	cv::Canny(result,cannyres,35,90);
	imshow("Result",result);
	imshow("Canny Result",cannyres);

	waitKey();

	return 0;
}


