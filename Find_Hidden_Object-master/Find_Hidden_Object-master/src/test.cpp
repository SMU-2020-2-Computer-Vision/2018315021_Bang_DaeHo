#include "opencv2/opencv.hpp"

typedef struct _Result
{
	double x = -1 ;
	double y = -1 ;
	int angle = 0 ;
	double scale = 1.0 ;
}Result ;

Result Test(cv::Mat input_template, cv::Mat input_background) ;
Result Find(cv::Mat input_template, cv::Mat input_background) ;


int main(int argc, char *argv[])
{
	if( argc != 3 )
	{
		printf("exe path_template path_background\n" );      // 옵션 문자열 출력
		return 1 ;
	}
	
	printf("template : %s\n", argv[1]) ;
	printf("back ground : %s\n", argv[2]) ;
	
	//read image
	//cv::Mat img = cv::imread("./img/find2.png") ;
	//cv::Mat tpl = cv::imread("./img/fish.png") ;
	cv::Mat tpl = cv::imread(argv[1]) ;
	cv::Mat img = cv::imread(argv[2]) ;
	
	//bgr to gray
	cv::Mat img_gray ;
	cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY) ;

	cv::Mat tpl_gray ;
	cv::cvtColor(tpl, tpl_gray, cv::COLOR_BGR2GRAY) ;

	//inverse	
	img_gray = 255 - img_gray ;
	tpl_gray = 255 - tpl_gray ;

	Result result = Test(tpl_gray, img_gray) ;
	
	/// Show Result
	cv::rectangle( img, cv::Point(result.x, result.y), cv::Point( result.x + tpl.cols , result.y + tpl.rows ), cv::Scalar(0,0,255), 2, 8, 0 );
	cv::imwrite( "result.png", img );

	return 0 ;
}

Result Test(cv::Mat input_template, cv::Mat input_background)
{
	/// Create the result matrix
	int result_cols =  input_background.cols - input_template.cols + 1;
	int result_rows = input_background.rows - input_template.rows + 1;

	cv::Mat result ;
	result.create( result_cols, result_rows, CV_32FC1 );

	/// Do the img and Normalize
	cv::matchTemplate( input_background, input_template, result, cv::TM_SQDIFF, input_template );

	/// Localizing the best match with minMaxLoc
	double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;
	cv::Point matchLoc;

	cv::minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat() );

	matchLoc = minLoc;

	Result ret ;
	ret.x = matchLoc.x ;
	ret.y = matchLoc.y ;

	return ret ;
	
}

Result Find(cv::Mat input_template, cv::Mat input_background)
{
	Result result ;

	// Alrogithm
	//..
	//
	
	return result ;
}



