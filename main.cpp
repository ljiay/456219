#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <malloc.h>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <math.h>
#define INVALID_PIXEL 0
using namespace cv;

typedef struct Image {
	int width;
	int height;
	unsigned char* leftGray;
	unsigned char* rightGray;
}Image;

typedef struct AggrCost {
	
};



Image image;

/// <summary>
/// census变换（5x5）
/// </summary>
/// <param name="input">输入图像像素</param>
/// <param name="output">输出census值</param>
/// <param name="width">图像宽</param>
/// <param name="height">图像高</param>
void censusTransform(const unsigned char* input, unsigned int* output, int width, int height)
{
	if (input == nullptr || output == nullptr || width <= 5 || height <= 5) {
		printf("Error in censusTransform!\n");
		return;
	}
	unsigned char centerGray = 0;
	//计算census值
	for (int i = 2; i < height - 2; i++) {
		for (int j = 2; j < width - 2; j++) {

			//中心像素的灰度值
			centerGray = input[i * width + j];

			// 遍历5x5窗口内邻域像素，计算census值（大于中心点的记为0，小于记为1）
			unsigned int censusValue = 0u;//后25位是census值
			for (int m = -2; m <= 2; m++) {
				for (int n = -2; n <= 2; n++) {
					censusValue <<= 1;
					unsigned char gray = input[(i + m) * width + j + n];
					if (gray < centerGray) {
						censusValue += 1;
					}
				}
			}

			output[i * width + j] = censusValue;
		}
	}
}

/// <summary>
/// 计算x与y的Hamming距离
/// </summary>
/// <param name="x"></param>
/// <param name="y"></param>
/// <returns></returns>
int hammingDistance(unsigned int x, unsigned int y)
{
	unsigned int distance = 0, val = x ^ y;//异或
	//统计val二进制串中1的个数
	while (val) {
		++distance;
		val &= val - 1;
	}
	return distance;
}

/// <summary>
/// 代价计算
/// </summary>
/// <param name="cost"></param>
/// <param name="minDisparity"></param>
/// <param name="maxDisparity"></param>
void computeCost(unsigned char* cost, int minDisparity, int maxDisparity) {
	const int width = image.width;
	const int height = image.height;
	const unsigned char* leftGray = image.leftGray;
	const unsigned char* rightGray = image.rightGray;
	const int pixelNum = height * width;	//像素数

	//census变换
	unsigned int* censusLeft = (unsigned int*)malloc(sizeof(unsigned int)  * pixelNum);
	unsigned int* censusRight = (unsigned int*)malloc(sizeof(unsigned int) * pixelNum);
	if (censusLeft == NULL || censusRight == NULL) {
		printf("Error in calculateCost: Malloc Failure\n");
		return;
	}
	censusTransform(leftGray, censusLeft, width, height);
	censusTransform(rightGray, censusRight, width, height);

	//计算初始代价
	int dispRange = maxDisparity - minDisparity;	//视差范围
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {

			unsigned int censusLeftValue = censusLeft[i * width + j];

			for (int m = minDisparity; m < maxDisparity; ++m) {
				int index = i * width * dispRange + j * dispRange + (m - minDisparity);

				if (j - m < 0 || j - m >= width) {
					cost[index] = 128;
					continue;
				}

				//计算初始匹配代价
				const unsigned int censusRightValue = censusRight[i * width + j - m];
				cost[index] = hammingDistance(censusLeftValue, censusRightValue);
			}

		}
	}

	free(censusLeft);
	free(censusRight);
}

//代价聚合--左右方向上
void costAggregationRow(unsigned char* initCost, unsigned char* aggrCost, int minDisparity, int maxDisparity, int P1, int initP2, bool leftToRight) {
	const int direction = leftToRight ? 1 : -1;
	const int width = image.width;
	const int height = image.height;
	const unsigned char* leftGray = image.leftGray;
	const unsigned char* rightGray = image.rightGray;
	int dispRange = maxDisparity - minDisparity;
	//int P2 = initP2;


	for (int i = 0; i < height; ++i) {
		//initCost和aggrCost大小均为w*h*d
		//w、d平面左下角元素的角标为：width*dispRange*i
		//w、d平面右下角元素的角标为：width*dispRange*(i+1)-dispRange
		int colHeadIndex = leftToRight ? (width * dispRange * i) : (width * dispRange * (i + 1) - dispRange);
		int pixelIndex = leftToRight ? (width * i) : (width * (i + 1) - 1);

		//第一个像素的聚合代价值等于它的初始代价值
		int lastPixelMinCost = 0xff;
		for (int d = 0; d < dispRange; ++d) {
			aggrCost[colHeadIndex + d] = initCost[colHeadIndex + d];
			if (initCost[colHeadIndex + d] < lastPixelMinCost) {
				lastPixelMinCost = initCost[colHeadIndex + d];
			}
		}
		
		int lastGray = image.leftGray[pixelIndex];
		int lastColHeadIndex = colHeadIndex;
		//计算第二个像素的角标
		colHeadIndex += direction * dispRange;
		pixelIndex += direction;


		//从第二个像素开始聚合
		for (int j = 1; j < width; ++j) {
			int nowGray = image.leftGray[pixelIndex];
			int P2 = initP2;
			if (lastGray != nowGray) {
				int P2 = initP2 / abs(lastGray - nowGray);
			}
			
			int nowPixelMinCost;		//记录当前像素最小代价
			for (int d = 0; d < dispRange; ++d) {
				int cost = initCost[colHeadIndex + d];
				int lr[4];
				lr[0] = aggrCost[lastColHeadIndex + d];
				if (d == 0) {
					lr[1] = 0xff + P1;
				}
				else {
					lr[1] = aggrCost[lastColHeadIndex + d - 1] + P1;
				}
				if (d == dispRange - 1) {
					lr[2] = 0xff + P1;
				}
				else {
					lr[2] = aggrCost[lastColHeadIndex + d + 1] + P1;
				}
				lr[3] = lastPixelMinCost + P2;
				
				//取lr的最小值
				int minLr = lr[0];
				for (int i = 1; i < 4; ++i) {
					if (minLr > lr[i]) {
						minLr = lr[i];
					}
				}

				aggrCost[colHeadIndex + d] = cost + minLr - lastPixelMinCost;

				if (d == 0) {
					nowPixelMinCost = aggrCost[colHeadIndex + d];
				}
				else if (aggrCost[colHeadIndex + d] < nowPixelMinCost) {
					nowPixelMinCost = aggrCost[colHeadIndex + d];
				}
			}

			lastPixelMinCost = nowPixelMinCost;

			//计算下一个像素的角标
			colHeadIndex += direction * dispRange;
			pixelIndex += dispRange;
			//更新上一个像素的灰度值
			lastGray = nowGray;
		}

	}
}

//视差计算--在视差范围内选择一个代价值最小的视差作为像素的最终视差
void computeDisparity(unsigned short* cost, unsigned short* disparity, int minDisparity, int maxDisparity) {
	int dispRange = maxDisparity - minDisparity;

	const int width = image.width;
	const int height = image.height;
	const unsigned char* leftGray = image.leftGray;
	const unsigned char* rightGray = image.rightGray;
	const int pixelNum = height * width;	//像素数

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			//遍历视差范围内的所有代价值，取代价值最小的视差作为像素的最终视差
			unsigned int min = 0xfffffff;
			unsigned int max = 0;
			unsigned int bestDisparity = 0;
			for (int m = minDisparity; m < maxDisparity; ++m) {
				int index = dispRange*(i * width + j ) + (m - minDisparity);
				if (cost[index] < min) {
					min = cost[index];
					bestDisparity = m;
				}
				if (cost[index] > max) {
					max = cost[index];
				}
			}
			
			//如果视差范围内的所有代价值均相同，则该像素无效
			if (max == min) {
				disparity[i * width + j] = INVALID_PIXEL;
			}
			else {//取代价值最小的视差作为像素的最终视差
				disparity[i * width + j] = bestDisparity;
			}
		}
	}
}

//加载图片
int loadImage(const char* leftImagePath,const char* rightImagePath) {
	Mat imgLeft = imread(leftImagePath, IMREAD_GRAYSCALE);
	Mat imgRight = imread(rightImagePath, IMREAD_GRAYSCALE);
	
	if (imgLeft.empty() || imgRight.empty()) {
		printf("Error in loadImage: Failed to load images\n");
		return -1;
	}
	if (imgLeft.rows != imgRight.rows || imgLeft.cols != imgRight.cols) {
		printf("Error in loadImage: Inconsistent image size\n");
		return -1;
	}

	//图像宽高、像素数
	image.width = imgLeft.cols;
	image.height = imgLeft.rows;
	int pixelNum = image.width * image.height;

	//将灰度信息保存到image结构体中
	image.leftGray = (unsigned char*)malloc(sizeof(unsigned char) * pixelNum);
	image.rightGray = (unsigned char*)malloc(sizeof(unsigned char) * pixelNum);
	if (image.leftGray == nullptr || image.rightGray == nullptr) {
		printf("Error in loadImage: Malloc Failure\n");
		return -1;
	}
	for (int i = 0; i < image.height; i++) {
		for (int j = 0; j < image.width; j++) {
			image.leftGray[i * image.width + j] = imgLeft.at<uchar>(i, j);
			image.rightGray[i * image.width + j] = imgRight.at<uchar>(i, j);
		}
	}
	/*
	namedWindow("Left", WINDOW_AUTOSIZE);
	namedWindow("Right", WINDOW_AUTOSIZE);
	imshow("Left", imgLeft);
	imshow("Right", imgRight);
	waitKey(0);
	destroyWindow("Left");
	destroyWindow("Right");
	*/
}

//显示视差图
void display(unsigned short* disparity) {
	const int width = image.width;
	const int height = image.height;
	Mat dispImage = cv::Mat(height, width, CV_8UC1);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			const float disp = disparity[i * width + j];//??????????????????
			if (disparity[i * width + j] == INVALID_PIXEL) {
				dispImage.data[i * width + j] = 0;
			}
			else {
				//dispImage.data[i * width + j] = static_cast<uchar>((disp - 0) / (64) * 255);
				dispImage.data[i * width + j] = disp * 4;
			}
		}
	}
	cv::imshow("视差图", dispImage);
	cv::waitKey(0);
}

//释放内存
void destroy() {
	if (image.leftGray) {
		free(image.leftGray);
	}
	if (image.rightGray) {
		free(image.rightGray);
	}
}

int main() {
	loadImage("G:\\Other\\image\\left1.png", "G:\\Other\\image\\right1.png");
	int minDisparity = 0;
	int maxDisparity = 64;
	unsigned char* initCost = (unsigned char*)malloc(sizeof(unsigned char) * image.height * image.width * (maxDisparity - minDisparity));
	unsigned char* aggrCost_1 = (unsigned char*)malloc(sizeof(unsigned char) * image.height * image.width * (maxDisparity - minDisparity));
	unsigned char* aggrCost_2 = (unsigned char*)malloc(sizeof(unsigned char) * image.height * image.width * (maxDisparity - minDisparity));
	unsigned short* aggrCost = (unsigned short*)malloc(sizeof(unsigned short) * image.height * image.width * (maxDisparity - minDisparity));
	unsigned short* disparity = (unsigned short*)malloc(sizeof(unsigned short) * image.height * image.width);
	//代价计算
	computeCost(initCost, minDisparity, maxDisparity);
	//代价聚合
	//costAggregationRow(initCost, aggrCost_1, minDisparity, maxDisparity, 2, 100, true);
	//costAggregationRow(initCost, aggrCost_2, minDisparity, maxDisparity, 2, 100, false);
	int size = (maxDisparity - minDisparity) * image.width * image.height;
	for (int i = 0; i < size; ++i) {
		//aggrCost[i] = aggrCost_1[i] +aggrCost_2[i];
		aggrCost[i] = initCost[i];
	}
	

	//视差计算
	computeDisparity(aggrCost, disparity, minDisparity, maxDisparity);

	display(disparity);
	destroy();
	return 0;
}