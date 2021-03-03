#include "Header.h"

//8 micron values
const int MASKX = 554;
const int MASKY = 79;
const int TEMPMASKX = 543;
const int TEMPMASKY = 71;

//15 micron values
//const int MASKX = 299;
//const int MASKY = 47;
//const int TEMPMASKX = 290;
//const int TEMPMASKY = 40;

void multiplyOwn(Mat& a, Mat& b, Mat& dst) {

	int event = 0;
	Mat aPrime = a.clone();
	Mat bPrime = b.clone();

	int height = a.rows;
	int width = a.cols;

	if (a.type() == 0 && b.type() == 0) {
		dst.create(height, width, CV_8UC1);
		event = 1;
	}
	else if (a.type() == 5 && b.type() == 0) {
		bPrime.convertTo(bPrime, CV_32FC1);
		dst.create(height, width, CV_32FC1);
		event = 2;
	}
	else if (a.type() == 0 && b.type() == 5) {
		aPrime.convertTo(aPrime, CV_32FC1);
		dst.create(height, width, CV_32FC1);
		event = 2;
	}
	else {
		dst.create(height, width, CV_32FC1);
		event = 2;
	}

	int i, j;

	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			if (event == 1) {
				dst.at<uchar>(i, j) = bPrime.at<uchar>(i, j) * aPrime.at<uchar>(i, j);
			}
			else if (event == 2) {
				dst.at<float>(i, j) = bPrime.at<float>(i, j) * aPrime.at<float>(i, j);
			}
			else {
				cout << "No Event Specified" << endl;
			}
		}
	}
}

void subtractOwn(Mat& a, Mat& b, Mat& dst) {

	int event = 0;
	Mat aPrime = a.clone();
	Mat bPrime = b.clone();

	int height = a.rows;
	int width = a.cols;

	if (a.type() == 0 && b.type() == 0) {
		dst.create(height, width, CV_8UC1);
		event = 1;
	}
	else if (a.type() == 5 && b.type() == 0) {
		bPrime.convertTo(bPrime, CV_32FC1);
		dst.create(height, width, CV_32FC1);
		event = 2;
	}
	else if (a.type() == 0 && b.type() == 5) {
		aPrime.convertTo(aPrime, CV_32FC1);
		dst.create(height, width, CV_32FC1);
		event = 2;
	}
	else {
		dst.create(height, width, CV_32FC1);
		event = 2;
	}

	int i, j;

	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			if (event == 1) {
				dst.at<uchar>(i, j) = aPrime.at<uchar>(i, j) - bPrime.at<uchar>(i, j);
			}
			else if (event == 2) {
				dst.at<float>(i, j) = aPrime.at<float>(i, j) - bPrime.at<float>(i, j);
			}
			else {
				cout << "No Event Specified" << endl;
			}
		}
	}
}

void nullifyNegatives(Mat& src, Mat& dst)
{
	int height = src.rows;
	int width = src.cols;
	int event = 0;

	if (src.type() == 5) {
		dst.create(height, width, CV_32FC1);
		event = 2;
	} else if (src.type() == 0) {
		dst.create(height, width, CV_8UC1);
		event = 1;
	}
	else {
		return;
	}

	if (event == 2) {
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				float temp = src.at<float>(i, j);
				if (temp < 0) {
					dst.at<float>(i, j) = 0;
				}
				else {
					dst.at<float>(i, j) = temp;
				}
			}
		}
	}
	else {
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				uchar temp = src.at<uchar>(i, j);
				if (temp < 0) {
					dst.at<uchar>(i, j) = 0;
				}
				else {
					dst.at<uchar>(i, j) = temp;
				}
			}
		}
	}
	return;
	cv::threshold(-src, dst, 0, 0, THRESH_TRUNC);
	dst = -dst;
	return;
}

string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

//Box Filter Kernel For Gray scale image with 8bit depth
//void box_filter_kernel_8u_c1(unsigned char* output, const int width, const int height, const size_t pitch, const int fWidth, const int fHeight)
//{
//	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
//	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
//
//	const int filter_offset_x = fWidth / 2;
//	const int filter_offset_y = fHeight / 2;
//
//	float output_value = 0.0f;
//
//	//Make sure the current thread is inside the image bounds
//	if (xIndex < width && yIndex < height)
//	{
//		//Sum the window pixels
//		for (int i = -filter_offset_x; i <= filter_offset_x; i++)
//		{
//			for (int j = -filter_offset_y; j <= filter_offset_y; j++)
//			{
//				//No need to worry about Out-Of-Range access. tex2D automatically handles it.
//				output_value += tex2D(tex8u, xIndex + i, yIndex + j);
//			}
//		}
//
//		//Average the output value
//		output_value /= (fWidth * fHeight);
//
//		//Write the averaged value to the output.
//		//Transform 2D index to 1D index, because image is actually in linear memory
//		int index = yIndex * pitch + xIndex;
//
//		output[index] = static_cast<unsigned char>(output_value);
//	}
//}

//all inputs must be 32F
void pickingPoints(Mat& q, Mat& guidedR1, Mat& dst, Mat EdgeImg2pt) {
	int i, j;
	int count = 0;
	int height = q.rows;
	int width = q.cols;
	dst.create(height, width, CV_32FC1);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			float temp = EdgeImg2pt.at<float>(i, j);
			if (temp >= 5) {
				float guidedPercentage = temp / 255;
				float guidedR1_percentage = 1.0 - guidedPercentage;

				dst.at<float>(i, j) = q.at<float>(i, j)    * guidedPercentage +
									  255 * guidedR1_percentage;
			}
			else {
				if (guidedR1.at<float>(i, j) > 128) {
					dst.at<float>(i, j) = 255.0;
				}
				else {
					dst.at<float>(i, j) = 0;
				}
			}
		} 
	}
}

//original edge guided filter

Mat guidedFilter(Mat& I, Mat& p, int r, float eps)
{
	Size ksize(2 * r + 1, 2 * r + 1);
	Mat Guide = I.clone();
	Mat Input = p.clone();

	//Step 1:
	//mean_I, mean_p, corr_I, corr_Ip
	Mat mean_p;
	Mat mean_I;
	Mat corr_Ip;
	Mat corr_I;

	boxFilter(Guide, mean_I, CV_32F, ksize);
	boxFilter(Input, mean_p, CV_32F, ksize);
	I.convertTo(Guide, CV_32FC1);
	p.convertTo(Input, CV_32FC1);

	Mat tmpIp;
	multiplyOwn(Guide, Input, tmpIp);
	boxFilter(tmpIp, corr_Ip, CV_32F, ksize);

	Mat_<double> mean_II, tmpII;
	tmpII = Guide.mul(Guide);
	boxFilter(tmpII, corr_I, CV_32F, ksize);


	//Step 2:
	//var_I  - variance of I in each local patch: the matrix Sigma in Eqn (14)
	//cov_Ip - covariance of (I, p) in each local patch 
	Mat var_I;
	Mat tmp_II;
	multiplyOwn(mean_I, mean_I, mean_II);
	subtractOwn(corr_I, mean_II, var_I);

	Mat cov_Ip;
	Mat mean_Ip;
	multiplyOwn(mean_I, mean_p, mean_Ip);
	cov_Ip = corr_Ip - mean_Ip;


	//Step 3:
	//compute a and b
	Mat a(Input.rows, Input.cols, CV_MAKETYPE(CV_32F, 1));
	Mat b(Input.rows, Input.cols, CV_32F);
	divide(cov_Ip, var_I + eps, a);

	Mat aMulmean_I;
	multiplyOwn(a, mean_I, aMulmean_I);
	b = mean_p - aMulmean_I;


	//Step 5:
	//find mean_a and mean_b
	Mat mean_a;
	Mat mean_b;
	boxFilter(a, mean_a, CV_32F, ksize);
	boxFilter(b, mean_b, CV_32F, ksize);
	Mat aI;
	multiplyOwn(mean_a, Guide, aI);

	Mat q, dst;
	q = aI + mean_b;

	//only taking q points of the edges
	q.convertTo(q, CV_8UC1);
	return q;
}

Mat noMeanGuidedFilter(Mat& I, Mat& p, int r, float eps )
{
	Size ksize(2 * r + 1, 2 * r + 1);
	Mat Guide = I.clone();
	Mat Input = p.clone();

	//Step 1:
	//mean_I, mean_p, corr_I, corr_Ip
	Mat mean_p;
	Mat mean_I;
	Mat corr_Ip;
	Mat corr_I;

	boxFilter(Guide, mean_I, CV_32F, ksize);
	boxFilter(Input, mean_p, CV_32F, ksize);
	I.convertTo(Guide, CV_32FC1);
	p.convertTo(Input, CV_32FC1);

	Mat tmpIp;
	multiplyOwn(Guide, Input, tmpIp);
	boxFilter(tmpIp, corr_Ip, CV_32F, ksize);

	Mat_<double> mean_II, tmpII;
	tmpII = Guide.mul(Guide);
	boxFilter(tmpII, corr_I, CV_32F, ksize);


	//Step 2:
	//var_I  - variance of I in each local patch: the matrix Sigma in Eqn (14)
	//cov_Ip - covariance of (I, p) in each local patch 

	Mat var_I;
	Mat tmp_II;
	multiplyOwn(mean_I, mean_I, mean_II);
	subtractOwn(corr_I, mean_II, var_I);

	Mat cov_Ip;
	Mat mean_Ip;
	multiplyOwn(mean_I, mean_p, mean_Ip);
	cov_Ip = corr_Ip - mean_Ip;


	//Step 3:
	//compute a and b

	Mat a(Input.rows, Input.cols, CV_MAKETYPE(CV_32F, 1));
	Mat b(Input.rows, Input.cols, CV_32F);
	divide(cov_Ip, var_I + eps, a);

	Mat aMulmean_I;
	multiplyOwn(a, mean_I, aMulmean_I);
	a = a * 1.0;
	b = mean_p - aMulmean_I;


	//getting output without mean
	Mat aI;
	multiplyOwn(a, Guide, aI);

	Mat q, dst;
	q = aI + b;
	q.convertTo(q, CV_8UC1);
	return q;
}

void correlationPadding(Mat& src, Mat& templ, Mat& dst ) {
	//catching if image cannot be read
    if (src.empty())
    {
        cout << "Can't read src image" << endl;
        return;
    }

    if (templ.empty())
    {
        cout << "Can't read template image" << endl;
        return;
    }

	Mat result;

    //make input image copy
    int result_cols = src.cols - templ.cols + 1;
    int result_rows = src.rows - templ.rows + 1;
    result.create(result_rows, result_cols, CV_32FC1);

    //Template Matching and Normalizing
    matchTemplate(src, templ, result, TM_CCORR);
    normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

    //Finding global max and min
    double minVal; double maxVal; Point minLoc; Point maxLoc;
    Point matchLoc;
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

    //get match location to be matchLoc
    matchLoc = maxLoc;
    cout << "Location for pic is: " << matchLoc << endl;

    //padding the obtained mask with zeroes
    Mat padded;
    copyMakeBorder(templ, padded, matchLoc.y, MASKY - matchLoc.y - TEMPMASKY, matchLoc.x, MASKX - TEMPMASKX - matchLoc.x, BORDER_CONSTANT, Scalar(0));
	padded.convertTo(padded, CV_8UC1);
	dst = padded;
}

void whitify(Mat& q) {
	int i, j;
	int count = 0;
	int height = q.rows;
	int width = q.cols;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (q.at<float>(i, j) > 1) {
				q.at<float>(i, j) = 255;
			}
		}
	}

}

Mat edgeGuidedFilter(Mat& I, Mat& p, int r, float eps)
{
	Mat imgDia, results_32, resultsR2_32, imgDia_32, dst;
	Mat guidedR1 = guidedFilter(I, p, 2, 0.1);

	//thickening canny edge by 1
	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	dilate(guidedR1, imgDia, kernel);

	//Getting guided 2pt edge
	guidedR1.convertTo(results_32, CV_32FC1);
	imgDia.convertTo(imgDia_32, CV_32FC1);
	whitify(imgDia_32);

	Mat guidedEdge2pt = imgDia_32 - results_32;

	//Nullifying edges
	Mat nullifiedGuidedEdge2pt;
	nullifyNegatives(guidedEdge2pt, nullifiedGuidedEdge2pt);

	//only taking q points of the edges
	Mat guidedR2 = guidedFilter(I, p, r, eps);
	guidedR2.convertTo(resultsR2_32, CV_32FC1);
	pickingPoints(resultsR2_32, results_32, dst, nullifiedGuidedEdge2pt);
	dst.convertTo(dst, CV_8UC1);
	return dst;
}

Mat guidedFilter_color(Mat& I, Mat& p, int r, float eps)
{
	Size ksize(2 * r + 1, 2 * r + 1);
	Mat mean_p;
	vector<Mat> bgr(3);
	split(I, bgr);
	Mat mean_b;
	Mat mean_g;
	Mat mean_r;
	boxFilter(bgr[0], mean_b, CV_32F, ksize);
	boxFilter(bgr[1], mean_g, CV_32F, ksize);
	boxFilter(bgr[2], mean_r, CV_32F, ksize);
	boxFilter(p, mean_p, -1, ksize);

	// covariance of (I, p) in each local patch 

	Mat cov_bp;
	Mat cov_gp;
	Mat cov_rp;

	{
		Mat tmp_bp;
		Mat mean_bp;
		multiply(bgr[0], p, tmp_bp);
		boxFilter(tmp_bp, mean_bp, CV_32F, ksize);
		multiply(mean_b, mean_p, tmp_bp);
		cov_bp = mean_bp - tmp_bp;
	}
	{
		Mat tmp_gp;
		Mat mean_gp;
		multiply(bgr[1], p, tmp_gp);
		boxFilter(tmp_gp, mean_gp, CV_32F, ksize);
		multiply(mean_g, mean_p, tmp_gp);
		cov_gp = mean_gp - tmp_gp;
	}
	{
		Mat tmp_rp;
		Mat mean_rp;
		multiply(bgr[2], p, tmp_rp);
		boxFilter(tmp_rp, mean_rp, CV_32F, ksize);
		multiply(mean_r, mean_p, tmp_rp);
		cov_rp = mean_rp - tmp_rp;
	}

	// variance of I in each local patch: the matrix Sigma in Eqn (14).
	Mat var_bb;
	Mat var_bg;
	Mat var_br;
	Mat var_gg;
	Mat var_gr;
	Mat var_rr;

	{
		Mat tmp_bb;
		Mat mean_bb;
		multiply(bgr[0], bgr[0], tmp_bb);
		boxFilter(tmp_bb, mean_bb, CV_32F, ksize);
		multiply(mean_b, mean_b, tmp_bb);
		var_bb = mean_bb - tmp_bb;
	}
	{
		Mat tmp_bg;
		Mat mean_bg;
		multiply(bgr[0], bgr[1], tmp_bg);
		boxFilter(tmp_bg, mean_bg, CV_32F, ksize);
		multiply(mean_b, mean_g, tmp_bg);
		var_bg = mean_bg - tmp_bg;
	}
	{
		Mat tmp_br;
		Mat mean_br;
		multiply(bgr[0], bgr[2], tmp_br);
		boxFilter(tmp_br, mean_br, CV_32F, ksize);
		multiply(mean_b, mean_r, tmp_br);
		var_br = mean_br - tmp_br;
	}
	{
		Mat tmp_gg;
		Mat mean_gg;
		multiply(bgr[1], bgr[1], tmp_gg);
		boxFilter(tmp_gg, mean_gg, CV_32F, ksize);
		multiply(mean_g, mean_g, tmp_gg);
		var_gg = mean_gg - tmp_gg;
	}
	{
		Mat tmp_gr;
		Mat mean_gr;
		multiply(bgr[1], bgr[2], tmp_gr);
		boxFilter(tmp_gr, mean_gr, CV_32F, ksize);
		multiply(mean_g, mean_r, tmp_gr);
		var_gr = mean_gr - tmp_gr;
	}
	{
		Mat tmp_rr;
		Mat mean_rr;
		multiply(bgr[2], bgr[2], tmp_rr);
		boxFilter(tmp_rr, mean_rr, CV_32F, ksize);
		multiply(mean_r, mean_r, tmp_rr);
		var_rr = mean_rr - tmp_rr;
	}

	// compute a and b
	Mat A_b(p.rows, p.cols, CV_MAKETYPE(CV_32F, 1));
	Mat A_g(p.rows, p.cols, CV_MAKETYPE(CV_32F, 1));
	Mat A_r(p.rows, p.cols, CV_MAKETYPE(CV_32F, 1));
	Mat B(p.rows, p.cols, CV_32F);

	for (int row = 0; row < p.rows; ++row)
	{
		for (int col = 0; col < p.cols; ++col)
		{
			Mat cov(3, 1, CV_32F);
			Mat a(3, 1, CV_32F);
			Mat sigma(3, 3, CV_32F);

			cov.at<float>(0, 0) = cov_bp.at<float>(row, col);
			cov.at<float>(1, 0) = cov_gp.at<float>(row, col);
			cov.at<float>(2, 0) = cov_rp.at<float>(row, col);
			sigma.at<float>(0, 0) = var_bb.at<float>(row, col);
			sigma.at<float>(1, 0) = var_bg.at<float>(row, col);
			sigma.at<float>(2, 0) = var_br.at<float>(row, col);
			sigma.at<float>(0, 1) = var_bg.at<float>(row, col);
			sigma.at<float>(1, 1) = var_gg.at<float>(row, col);
			sigma.at<float>(2, 1) = var_gr.at<float>(row, col);
			sigma.at<float>(0, 2) = var_br.at<float>(row, col);
			sigma.at<float>(1, 2) = var_gr.at<float>(row, col);
			sigma.at<float>(2, 2) = var_rr.at<float>(row, col);
			sigma += eps * Mat::eye(3, 3, CV_32F);
			solve(sigma, cov, a, DECOMP_CHOLESKY);

			int idx[3] = { row, col, 0 };
			A_b.at<float>(idx) = a.at<float>(0, 0);
			A_g.at<float>(idx) = a.at<float>(1, 0);
			A_r.at<float>(idx) = a.at<float>(2, 0);
			B.at<float>(row, col) = mean_p.at<float>(row, col)
				- a.at<float>(0, 0) * mean_b.at<float>(row, col) - a.at<float>(1, 0) * mean_g.at<float>(row, col) - a.at<float>(2, 0) * mean_r.at<float>(row, col);
		}
	}

	Mat A_bb, A_gg, A_rr, BB;
	boxFilter(A_b, A_bb, CV_32F, ksize);
	boxFilter(A_g, A_gg, CV_32F, ksize);
	boxFilter(A_r, A_rr, CV_32F, ksize);
	boxFilter(B, BB, CV_32F, ksize);
	multiply(A_bb, bgr[0], A_b);
	multiply(A_gg, bgr[1], A_g);
	multiply(A_rr, bgr[2], A_r);

	Mat q;
	q = A_b + A_g + A_r + BB;
#ifdef _DEBUG
	cout << "Press any key to continue..." << endl;
	imshow("B", bgr[0]);
	imshow("G", bgr[1]);
	imshow("R", bgr[2]);
	waitKey(0);
#endif

	return q;
}

Mat guidedFilter_gray(Mat& I, Mat& p, int r, float eps)
{
	if (I.channels() == 3)
	{
		return guidedFilter_color(I, p, r, eps);
	}
	else if (I.channels() == 1)
	{
		return guidedFilter_gray(I, p, r, eps);
	}
	return Mat::Mat();
}

//Clean Guided Edge filtering
int main() {
	//Reading images
	//Mat input = imread("Resources/15 Micron Images/ASE15um/MedianMask/PaddedMask.bmp", IMREAD_GRAYSCALE);
	//Mat input   = imread("Resources/15 Micron Images/ASE15um/Angle 3 Reduce Border 0/42/FlexiGoldFinger_CurMask.jpg", IMREAD_GRAYSCALE);
	vector<int> arr{ 1,2,21,22,41,42,61,62 };

	for (int i = 0; i < 8; i++) {
		Mat masktempl = imread("Resources/8 Micron Images/Median Stack Images/maskTemplate.bmp", IMREAD_GRAYSCALE);
		Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
		erode(masktempl, masktempl, kernel);
		Mat guide = imread("Resources/8 Micron Images/Angle 3 Reduce Border 0/"+ to_string(arr[i])+"/FlexiGoldFinger_RegisteredInspectionImage.bmp", IMREAD_GRAYSCALE);
		//Mat guide = imread("Resources/8 Micron Images/Angle 3 Reduce Border 0/21/FlexiGoldFinger_RegisteredInspectionImage.jpg", IMREAD_GRAYSCALE);
		//int rows = input.rows;
		//int cols = input.cols;
		//cout << "dimensions x" << rows << endl;
		//cout << "dimensions y" << cols << endl;

		//catching if image cannot be read
		if (masktempl.empty())
		{
			cout << "Can't read input image" << endl;
			return EXIT_FAILURE;
		}
		if (guide.empty())
		{
			cout << "Can't read guide image" << endl;
			return EXIT_FAILURE;
		}

		Mat input;
		correlationPadding(guide, masktempl, input);

		//////Trackbar
		int r1 = 2;
		int x1 = 1;
		int r2 = 2;
		int x2 = 1;
		int threshold_L = 205;
		int threshold_U = 255;

		//namedWindow("Trackbar", (1500, 1000));
		//createTrackbar("r1", "Trackbar", &r1, 50);
		//createTrackbar("x1", "Trackbar", &x1, 20);
		//createTrackbar("r2", "Trackbar", &r2, 50);
		//createTrackbar("x2", "Trackbar", &x2, 20);
		//createTrackbar("Image Upper Threshold", "Trackbar", &threshold_U, 255);
		//createTrackbar("Image Lower Threshold", "Trackbar", &threshold_L, 255);

		//while (true) {
			Mat resultsOpenCV;
			Mat results;
			Mat resultsOpenCV_32;
			Mat results_32;
			Mat resultsSource;
			Mat imgCanny;
			Mat imgDia;
			Mat imgDia_32;
			Mat guidedEdge2pt;
			Mat guidedWeights;

			//running opencv guided filter first
			//cv::ximgproc::guidedFilter(guide, input, resultsOpenCV, r1, pow(10, -x1));
			results = guidedFilter(guide, input, r1, pow(10, -x1));
			//Canny(results, imgCanny, threshold_L, threshold_U);


			resultsSource = edgeGuidedFilter(guide, input, r2, pow(10, -x2));

			//concantenate filtered results vertically
			Mat a, b, c, d, e, f, g, h, j, guidedEdge2pt_8, nullifiedGuidedEdge2pt_8;

			resize(input, a, Size(), 1.0, 1.0);
			resize(guide, b, Size(), 1.0, 1.0);
			//resize(resultsOpenCV, c, Size(), 2, 2);
			resize(results, c, Size(), 1.0, 1.0);
			resize(resultsSource, d, Size(), 1.0, 1, 3);
			h = resultsSource - guide;
			nullifyNegatives(h, h);
			resize(h, h, Size(), 1.0, 1.0);

			j = input - guide;
			nullifyNegatives(j, j);
			normalize(j, j, 0, 255, NORM_MINMAX, -1, Mat());
			resize(j, j, Size(), 1.0, 1.0);


			Mat imgArray[] = { b, c, d, h, a, j};
			Mat dst;
			vconcat(imgArray, 6, dst);
			//imshow("Results", dst);
			imwrite("Resources/8 Micron Images/Angle 3 Reduce Border 1/Results-EdgeGuidedFilterRefined/results" + to_string(arr[i]) + ".bmp", dst);
		//	waitKey(40);
		//}
	}
}

////Clean Guided Canny Code
//int main() {
//	//Reading images
//	//Mat input = imread("Resources/15 Micron Images/ASE15um/MedianMask/PaddedMask.bmp", IMREAD_GRAYSCALE);
//	//Mat input   = imread("Resources/15 Micron Images/ASE15um/Angle 3 Reduce Border 0/42/FlexiGoldFinger_CurMask.jpg", IMREAD_GRAYSCALE);
//	vector<int> arr{ 1,2,21,22,41,42,61,62 };
//
//	for (int i = 0; i < 8; i++) {
//		Mat masktempl = imread("Resources/8 Micron Images/Median Stack Images/maskTemplate.bmp", IMREAD_GRAYSCALE);
//		Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
//		erode(masktempl, masktempl, kernel);
//		Mat guide = imread("Resources/8 Micron Images/Angle 3 Reduce Border 0/" + to_string(arr[i]) + "/FlexiGoldFinger_RegisteredInspectionImage.bmp", IMREAD_GRAYSCALE);
//		//Mat guide = imread("Resources/8 Micron Images/Angle 3 Reduce Border 0/21/FlexiGoldFinger_RegisteredInspectionImage.jpg", IMREAD_GRAYSCALE);
//		//int rows = input.rows;
//		//int cols = input.cols;
//		//cout << "dimensions x" << rows << endl;
//		//cout << "dimensions y" << cols << endl;
//
//		//catching if image cannot be read
//		if (masktempl.empty())
//		{
//			cout << "Can't read input image" << endl;
//			return EXIT_FAILURE;
//		}
//		if (guide.empty())
//		{
//			cout << "Can't read guide image" << endl;
//			return EXIT_FAILURE;
//		}
//
//		Mat input;
//		correlationPadding(guide, masktempl, input);
//
//		//////Trackbar
//		int r1 = 2;
//		int x1 = 1;
//		int r2 = 4;
//		int x2 = 1;
//		int threshold_L = 205;
//		int threshold_U = 255;
//
//		//namedWindow("Trackbar", (1500, 1000));
//		//createTrackbar("r1", "Trackbar", &r1, 50);
//		//createTrackbar("x1", "Trackbar", &x1, 20);
//		//createTrackbar("r2", "Trackbar", &r2, 50);
//		//createTrackbar("x2", "Trackbar", &x2, 20);
//		//createTrackbar("Image Upper Threshold", "Trackbar", &threshold_U, 255);
//		//createTrackbar("Image Lower Threshold", "Trackbar", &threshold_L, 255);
//
//		//while (true) {
//		Mat resultsOpenCV;
//		Mat results;
//		Mat resultsOpenCV_32;
//		Mat results_32;
//		Mat resultsSource;
//		Mat imgCanny;
//		Mat imgDia;
//		Mat imgDia_32;
//		Mat guidedEdge2pt;
//		Mat guidedWeights;
//
//		//running opencv guided filter first
//		//cv::ximgproc::guidedFilter(guide, input, resultsOpenCV, r1, pow(10, -x1));
//		results = guidedFilter(guide, input, r1, pow(10, -x1));
//		//Canny(results, imgCanny, threshold_L, threshold_U);
//
//		//thickening canny edge by 1
//		kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
//		dilate(results, imgDia, kernel);
//
//		//Getting guided 2pt edge
//		results.convertTo(results_32, CV_32FC1);
//		imgDia.convertTo(imgDia_32, CV_32FC1);
//		whitify(imgDia_32);
//
//		guidedEdge2pt = imgDia_32 - results_32;
//
//		//Nullifying edges
//		Mat nullifiedGuidedEdge2pt;
//		nullifyNegatives(guidedEdge2pt, nullifiedGuidedEdge2pt);
//
//		//getting weights
//		resultsSource = edgeGuidedFilter(guide, input, results_32, r2, pow(10, -x2), nullifiedGuidedEdge2pt);
//
//		//concantenate filtered results vertically
//		Mat a, b, c, d, e, f, g, h, j, guidedEdge2pt_8, nullifiedGuidedEdge2pt_8;
//		guidedEdge2pt.convertTo(guidedEdge2pt_8, CV_8UC1);
//		nullifiedGuidedEdge2pt.convertTo(nullifiedGuidedEdge2pt_8, CV_8UC1);
//
//		resize(input, a, Size(), 1.0, 1.0);
//		resize(guide, b, Size(), 1.0, 1.0);
//		//resize(resultsOpenCV, c, Size(), 2, 2);
//		resize(results, c, Size(), 1.0, 1.0);
//
//		//resize(imgCanny, d, Size(), 1.0, 1.0);
//		resize(imgDia, e, Size(), 1.0, 1.0);
//		resize(nullifiedGuidedEdge2pt_8, f, Size(), 1.0, 1.0);
//
//		resize(resultsSource, g, Size(), 1.0, 1, 3);
//		h = resultsSource - guide;
//		nullifyNegatives(h, h);
//		resize(h, h, Size(), 1.0, 1.0);
//
//		j = input - guide;
//		nullifyNegatives(j, j);
//		normalize(j, j, 0, 255, NORM_MINMAX, -1, Mat());
//		resize(j, j, Size(), 1.0, 1.0);
//
//
//		Mat imgArray[] = { b, c, e, f, a, j,  g, h };
//		Mat dst;
//		vconcat(imgArray, 8, dst);
//		//imshow("Results", dst);
//		imwrite("Resources/8 Micron Images/Angle 3 Reduce Border 1/Results-EdgeGuidedFilter/results" + to_string(arr[i]) + ".bmp", dst);
//		//Mat imgArray[] = { a, b, c, d};
//		//Mat dst;
//		//vconcat(imgArray, 4, dst);
//		//imshow("Results", dst);
//
//
//		/* imshow("1px guided results", f);
//		imshow("Canny", a);
//		imshow("Dia", b);
//		imshow("2pt edge", c);
//		imshow("Nullified 2pt edge", g);
//		imshow("result source", d);
//		imshow("Results Source - guide", e);*/
//
//		//	waitKey(40);
//		//}
//	}
//}

//Guided Canny Attempt#1
//int main() {
//	//Reading images
//	Mat input = imread("Resources/15 Micron Images/ASE15um/MedianMask/PaddedMask.bmp", IMREAD_GRAYSCALE);
//	Mat guide = imread("Resources/15 Micron Images/ASE15um/Angle 3 Reduce Border 0/1/FlexiGoldFinger_RegisteredInspectionImage.jpg", IMREAD_GRAYSCALE);
//	Mat edgeImg = imread("Resources/15 Micron Images/ASE15um/EdgeMasks/2ptEdgeMask.bmp", IMREAD_GRAYSCALE);
//	int numEdges;
//
//	//catching if image cannot be read
//	if (input.empty())
//	{
//		cout << "Can't read input image" << endl;
//		return EXIT_FAILURE;
//	}
//	if (guide.empty())
//	{
//		cout << "Can't read guide image" << endl;
//		return EXIT_FAILURE;
//	}
//	if (edgeImg.empty())
//	{
//		cout << "Can't read edge image" << endl;
//		return EXIT_FAILURE;
//	}
//
//	////////Trackbar
//	int r1 = 1;
//	int x1 = 1;
//	int threshold_L = 78;
//	int threshold_U = 213;
//
//	namedWindow("Trackbar", (1500, 1000));
//	createTrackbar("r1", "Trackbar", &r1, 50);
//	createTrackbar("x1", "Trackbar", &x1, 20);
//	createTrackbar("Image Upper Threshold", "Trackbar", &threshold_U, 255);
//	createTrackbar("Image Lower Threshold", "Trackbar", &threshold_L, 255);
//
//	while (true) {
//		Mat resultsOpenCV;
//		Mat resultsOpenCV_32;
//		Mat resultsSource;
//		Mat imgCanny;
//		Mat imgDia;
//		Mat imgDia_32;
//		Mat guidedEdge2pt;
//		Mat guidedWeights;
//
//		//comparing with opencv guided filter results
//		cv::ximgproc::guidedFilter(guide, input, resultsOpenCV, 1, pow(10, -1));
//		
//		Canny(resultsOpenCV, imgCanny, threshold_L, threshold_U);
//
//			//thickening canny edge by 1
//		Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
//		dilate(imgCanny, imgDia, kernel);
//		
//			//Getting guided 2pt edge
//		resultsOpenCV.convertTo(resultsOpenCV_32, CV_32FC1);
//		imgDia.convertTo(imgDia_32, CV_32FC1);
//		//cout << "Results OpenCV: " << resultsOpenCV_32.type() << endl;
//		//cout << "Img Dialate: " << imgDia_32.type() << endl;
//		guidedEdge2pt = imgDia_32 - resultsOpenCV_32; //guidedEdge pt is having -ve values
//		//cout << "Guided edge 2pt: " << guidedEdge2pt.type() << endl;
//		//cout << "guided2pt" << endl << guidedEdge2pt << endl << endl;
//		Mat nullifiedGuidedEdge2pt;
//
//
//		nullifyNegatives(guidedEdge2pt, nullifiedGuidedEdge2pt);
//		//cout << "guided2ptNullified" << endl << nullifiedGuidedEdge2pt << endl << endl;
//		
//			//locating edges into an array
//		vector<Point> edgesArr = edgePointLocator(nullifiedGuidedEdge2pt, &numEdges);
//
//			//getting weights
//		normalize(nullifiedGuidedEdge2pt, guidedWeights, 0, 1, NORM_MINMAX, -1, Mat());
//		//cout << "Guided Weight Weights: " << guidedWeights.type() << endl;
//		resultsSource = edgeGuidedFilter(guide, input, r1, pow(10, -x1), edgesArr, numEdges, guidedWeights);
//		//cout << "results Source: " << endl << resultsSource << endl << endl;
//
//		//concantenate filtered results vertically
//		Mat a, b, c, d, e, f, g;
//		resize(resultsOpenCV, f, Size(), 2, 2);
//		resize(imgCanny, a, Size(), 2, 2);
//		resize(imgDia, b, Size(), 2, 2);
//		resize(guidedEdge2pt, c, Size(), 1, 1);
//		resize(nullifiedGuidedEdge2pt, g, Size(), 1, 1);
//		resize(resultsSource, d, Size(), 2, 2);
//		resize(resultsSource - guide, e, Size(), 2, 2);
//
//		//imwrite("Resources/guidedEdge2pt.bmp", guidedEdge2pt);
//		//imwrite("Resources/dialated.bmp", imgDia);
//		//imwrite("Resources/guidedMask1.bmp", resultsOpenCV);
//
//
//		//Mat imgArray[] = { a, b, c, d, e };
//		//Mat dst;
//		//vconcat(imgArray, 5, dst);
//		//imshow("Results", dst);
//		imshow("1px guided results", f);
//		imshow("Canny", a);
//		imshow("Dia", b);
//		imshow("2pt edge", c);
//		imshow("Nullified 2pt edge", g);
//		imshow("result source", d);
//		imshow("Results Source - guide", e);
//
//		waitKey(100000);
//	}
//}

//guided Canny Guided
//int main() {
//	//Reading images
//	Mat input = imread("Resources/15 Micron Images/ASE15um/MedianMask/PaddedMask.bmp", IMREAD_GRAYSCALE);
//	Mat guide = imread("Resources/15 Micron Images/ASE15um/Angle 3 Reduce Border 0/1/FlexiGoldFinger_RegisteredInspectionImage.jpg", IMREAD_GRAYSCALE);
//	Mat edgeImg = imread("Resources/15 Micron Images/ASE15um/EdgeMasks/2ptEdgeMask.bmp", IMREAD_GRAYSCALE);
//
//	int numEdges;
//
//	//catching if image cannot be read
//	if (input.empty())
//	{
//		cout << "Can't read input image" << endl;
//		return EXIT_FAILURE;
//	}
//	if (guide.empty())
//	{
//		cout << "Can't read guide image" << endl;
//		return EXIT_FAILURE;
//	}
//	if (edgeImg.empty())
//	{
//		cout << "Can't read edge image" << endl;
//		return EXIT_FAILURE;
//	}
//
//	////////Trackbar
//	int r1 = 1;
//	int x1 = 1;
//	int r2 = 1;
//	int x2 = 1;
//
//	namedWindow("Trackbar", (1500, 1000));
//	createTrackbar("Image Upper Threshold", "Trackbar", &r1, 50);
//	createTrackbar("Image Lower Threshold", "Trackbar", &x1, 20);
//
//	//locating edges into an array
//	vector<Point> edgesArr = edgePointLocator(edgeImg, &numEdges);
//
//	while (true) {
//		Mat resultsOpenCV;
//		Mat resultsSource;
//
//		//comparing with opencv guided filter results
//		cv::ximgproc::guidedFilter(guide, input, resultsOpenCV, r1, pow(10, -x1));
//		resultsSource = edgeGuidedFilter(guide, input, r1, pow(10, -x1), edgesArr, numEdges);
//		//cout << "reference guided filter = " << endl << " " << resultsOpenCV << endl << endl;
//
//		//concantenate filtered results vertically
//		Mat a, b, c, d;
//		resize(resultsOpenCV, a, Size(), 3, 3);
//		resize(resultsSource, b, Size(), 3, 3);
//		resize(resultsOpenCV - guide, c, Size(), 3, 3);
//		resize(resultsSource - guide, d, Size(), 3, 3);
//
//		Mat imgArray[] = { a, b, c, d };
//		Mat dst;
//		vconcat(imgArray, 4, dst);
//		imshow("Results", dst);
//		//imshow("Results Source", a);
//		//imshow("Results Opencv", b);
//
//		waitKey(30);
//	}
//}

//int main() {
//	//Reading images
//	Mat input = imread("Resources/15 Micron Images/ASE15um/MedianMask/PaddedMask.bmp", IMREAD_GRAYSCALE);
//	Mat guide = imread("Resources/15 Micron Images/ASE15um/Angle 3 Reduce Border 0/1/FlexiGoldFinger_RegisteredInspectionImage.jpg", IMREAD_GRAYSCALE);
//	Mat edgeImg = imread("Resources/15 Micron Images/ASE15um/EdgeMasks/2ptEdgeMask.bmp", IMREAD_GRAYSCALE);
//	//Mat input = imread("Resources/InputImage.bmp", IMREAD_GRAYSCALE);
//	//Mat guide = imread("Resources/GuideImage.bmp", IMREAD_GRAYSCALE);
//	//Mat smootest = imread("Resources/GuideImage.bmp", IMREAD_GRAYSCALE);
//	//Mat edgeImg3 = imread("Resources/edge3px.bmp", IMREAD_GRAYSCALE);
//	//Mat edgeImg2 = imread("Resources/edge2px.bmp", IMREAD_GRAYSCALE);
//	//Mat edgeImg1 = imread("Resources/edge1px.bmp", IMREAD_GRAYSCALE);
//	int numEdges;
//
//	//catching if image cannot be read
//	if (input.empty())
//	{
//		cout << "Can't read input image" << endl;
//		return EXIT_FAILURE;
//	}
//	if (guide.empty())
//	{
//		cout << "Can't read guide image" << endl;
//		return EXIT_FAILURE;
//	}
//	if (edgeImg.empty())
//	{
//		cout << "Can't read edge image" << endl;
//		return EXIT_FAILURE;
//	}
//
//	////////Trackbar
//	int r1 = 1;
//	int x1 = 1;
//	int r2 = 1;
//	int x2 = 1;
//
//	namedWindow("Trackbar", (1500, 1000));
//	createTrackbar("Image Upper Threshold", "Trackbar", &r1, 50);
//	createTrackbar("Image Lower Threshold", "Trackbar", &x1, 20);
//
//	//locating edges into an array
//	vector<Point> edgesArr = edgePointLocator(edgeImg, &numEdges);
//
//	while (true) {
//		Mat resultsOpenCV;
//		Mat resultsSource;
//
//		//comparing with opencv guided filter results
//		cv::ximgproc::guidedFilter(guide, input, resultsOpenCV, r1, pow(10, -x1));
//		resultsSource = edgeGuidedFilter(guide, input, r1, pow(10, -x1), edgesArr, numEdges);
//		//cout << "reference guided filter = " << endl << " " << resultsOpenCV << endl << endl;
//
//		//concantenate filtered results vertically
//		Mat a, b, c, d;
//		resize(resultsOpenCV, a, Size(), 3, 3);
//		resize(resultsSource, b, Size(), 3, 3);
//		resize(resultsOpenCV - guide, c, Size(), 3, 3);
//		resize(resultsSource - guide, d, Size(), 3, 3);
//
//		Mat imgArray[] = { a, b, c, d };
//		Mat dst;
//		vconcat(imgArray, 4, dst);
//		imshow("Results", dst);
//		//imshow("Results Source", a);
//		//imshow("Results Opencv", b);
//
//		waitKey(30);
//	}
////}