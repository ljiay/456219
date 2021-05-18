#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#define INVALID_PIXEL 0
#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))
using namespace cv;

typedef struct Image {
	int width;
	int height;
	unsigned char* leftGray;
	unsigned char* rightGray;
}Image;
Image image;

unsigned char* initCost;	//初始代价
unsigned char* aggrCost_1;	
unsigned char* aggrCost_2;
unsigned char* aggrCost_3;
unsigned char* aggrCost_4;
unsigned short* aggrCost;	//聚合代价

unsigned short* disparity;	//视差
int minDisparity = 0;
int maxDisparity = 64;
int P1 = 10;
int initP2 = 150;

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
//加载图片
int loadImage(const char* leftImagePath, const char* rightImagePath) {
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
}


//计算初始代价
void computeCost() {
	const int width = image.width;
	const int height = image.height;
	const unsigned char* leftGray = image.leftGray;
	const unsigned char* rightGray = image.rightGray;
	const int pixelNum = height * width;	//像素数

	//census变换
	unsigned int* censusLeft = (unsigned int*)malloc(sizeof(unsigned int) * pixelNum);
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
					initCost[index] = 128;
					continue;
				}

				//计算初始匹配代价
				const unsigned int censusRightValue = censusRight[i * width + j - m];
				initCost[index] = hammingDistance(censusLeftValue, censusRightValue);
			}

		}
	}

	free(censusLeft);
	free(censusRight);
}

//代价聚合--左右方向上
void costAggregationRow(bool leftToRight) {
	const int direction = leftToRight ? 1 : -1;
	const int width = image.width;
	const int height = image.height;
	const unsigned char* leftGray = image.leftGray;
	int dispRange = maxDisparity - minDisparity;
	unsigned char* aggrCostRow = leftToRight ? aggrCost_1 : aggrCost_2;
	for (int i = 0; i < height; i++)
	{
		unsigned char* initCostNowPos = leftToRight ? initCost + i * width * dispRange :
			initCost + ((i + 1) * width - 1) * dispRange;
		unsigned char* aggrCostNowPos = leftToRight ? aggrCostRow + i * width * dispRange :
			aggrCostRow + ((i + 1) * width - 1) * dispRange;
		unsigned char* imgNowPos = leftToRight ? image.leftGray + i * width : image.leftGray + (i + 1) * width - 1;
		unsigned char* aggrCostLastPos, * imgLastPos;

		memcpy(aggrCostNowPos, initCostNowPos, dispRange * sizeof(unsigned char));

		unsigned int minAggrCostOfLastPos = 255;
		for (int d = 0; d < dispRange; d++)
			minAggrCostOfLastPos = min(minAggrCostOfLastPos, aggrCostNowPos[d]);

		for (int j = 1; j < width; j++)
		{
			aggrCostLastPos = aggrCostNowPos;
			imgLastPos = imgNowPos;
			initCostNowPos += direction * dispRange;
			aggrCostNowPos += direction * dispRange;
			imgNowPos += direction;

			unsigned int minAggrCostOfNowPos = 255;
			for (int d = 0; d < dispRange; d++)
			{
				int l1 = aggrCostLastPos[d];
				int l2 = d == 0 ? 255 + P1: aggrCostLastPos[d - 1] + P1;
				int l3 = d == dispRange - 1 ? 255 + P1 : aggrCostLastPos[d + 1] + P1;
				int l4 = minAggrCostOfLastPos + initP2 / (abs(*imgNowPos - *imgLastPos) + 1);
				aggrCostNowPos[d] = initCostNowPos[d] + min(min(l1, l2), min(l3, l4)) - minAggrCostOfLastPos;
				minAggrCostOfNowPos = min(minAggrCostOfNowPos, aggrCostNowPos[d]);
			}
			minAggrCostOfLastPos = minAggrCostOfNowPos;
		}
	}
}
void costAggregationCol(bool bottomToTop) {
	const int direction = bottomToTop ? 1 : -1;
	const int width = image.width;
	const int height = image.height;
	const unsigned char* leftGray = image.leftGray;
	int dispRange = maxDisparity - minDisparity;
	unsigned char* aggrCostCol = bottomToTop ? aggrCost_3 : aggrCost_4;
	for (int i = 0; i < width; i++)
	{
		unsigned char* initCostNowPos = bottomToTop ? initCost + i * dispRange :
			initCost + ((height - 1) * width + i) * dispRange;
		unsigned char* aggrCostNowPos = bottomToTop ? aggrCostCol + i * dispRange :
			aggrCostCol + ((height - 1) * width + i) * dispRange;
		unsigned char* imgNowPos = bottomToTop ? image.leftGray + i : image.leftGray + (height - 1) * width + i;
		unsigned char* aggrCostLastPos, * imgLastPos;

		memcpy(aggrCostNowPos, initCostNowPos, dispRange * sizeof(unsigned char));

		unsigned char minAggrCostOfLastPos = 255;
		for (int d = 0; d < dispRange; d++)
			minAggrCostOfLastPos = min(minAggrCostOfLastPos, aggrCostNowPos[d]);

		for (int j = 1; j < height; j++)
		{
			aggrCostLastPos = aggrCostNowPos;
			imgLastPos = imgNowPos;
			initCostNowPos += direction * width * dispRange;
			aggrCostNowPos += direction * width * dispRange;
			imgNowPos += direction * width;

			int P2 = *imgNowPos == *imgLastPos ? initP2 : initP2 / abs(*imgNowPos - *imgLastPos);
			unsigned char minAggrCostOfNowPos = 255;
			for (int d = 0; d < dispRange; d++)
			{
				int l1 = aggrCostLastPos[d];
				int l2 = d == 0 ? 255 + P1 : aggrCostLastPos[d - 1] + P1;
				int l3 = d == dispRange - 1 ? 255 + P1 : aggrCostLastPos[d + 1] + P1;
				int l4 = minAggrCostOfLastPos + P2;
				aggrCostNowPos[d] = initCostNowPos[d] + min(min(l1, l2), min(l3, l4)) - minAggrCostOfLastPos;
				minAggrCostOfNowPos = min(minAggrCostOfNowPos, aggrCostNowPos[d]);
			}
			minAggrCostOfLastPos = minAggrCostOfNowPos;
		}
	}
}

//视差计算--在视差范围内选择一个代价值最小的视差作为像素的最终视差
void computeDisparity() {
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
				int index = dispRange * (i * width + j) + (m - minDisparity);
				if (aggrCost[index] < min) {
					min = aggrCost[index];
					bestDisparity = m;
				}
				if (aggrCost[index] > max) {
					max = aggrCost[index];
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

//显示视差图
void display() {
	const int width = image.width;
	const int height = image.height;
	Mat dispImage = cv::Mat(height, width, CV_8UC1);
	int dispRange = maxDisparity - minDisparity;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			const unsigned short disp = disparity[i * width + j];
			if (disparity[i * width + j] == INVALID_PIXEL) {
				dispImage.data[i * width + j] = 0;
			}
			else {
				//将disp映射到区间[0,255]
				dispImage.data[i * width + j] = static_cast<uchar>(((float)disp - minDisparity) / (dispRange) * 255);
			}
		}
	}
	cv::imshow("视差图", dispImage);
	cv::waitKey(0);
}
//初始化
void init(const char* leftImagePath, const char* rightImagePath) {
	loadImage(leftImagePath, rightImagePath);
	initCost = (unsigned char*)malloc(sizeof(unsigned char) * image.height * image.width * (maxDisparity - minDisparity));
	aggrCost_1 = (unsigned char*)malloc(sizeof(unsigned char) * image.height * image.width * (maxDisparity - minDisparity));
	aggrCost_2 = (unsigned char*)malloc(sizeof(unsigned char) * image.height * image.width * (maxDisparity - minDisparity));
	aggrCost_3 = (unsigned char*)malloc(sizeof(unsigned char) * image.height * image.width * (maxDisparity - minDisparity));
	aggrCost_4 = (unsigned char*)malloc(sizeof(unsigned char) * image.height * image.width * (maxDisparity - minDisparity));
	aggrCost = (unsigned short*)malloc(sizeof(unsigned short) * image.height * image.width * (maxDisparity - minDisparity));
	disparity = (unsigned short*)malloc(sizeof(unsigned short) * image.height * image.width);
}
//释放内存
void destroy() {
	if (image.leftGray) {
		free(image.leftGray);
	}
	if (image.rightGray) {
		free(image.rightGray);
	}
	free(initCost);
	free(aggrCost);
	free(aggrCost_1);
	free(aggrCost_2);
	free(aggrCost_3);
	free(aggrCost_4);
	free(disparity);
}

int main() {
	init("G:\\Other\\image\\left1.png", "G:\\Other\\image\\right1.png");
	//代价计算
	computeCost();
	//代价聚合
	costAggregationRow(true);
	costAggregationRow(false);
	costAggregationCol(true);
	costAggregationCol(false);
	int size = (maxDisparity - minDisparity) * image.width * image.height;
	for (int i = 0; i < size; ++i) {
		aggrCost[i] = aggrCost_1[i] + aggrCost_2[i] + aggrCost_3[i] + aggrCost_4[i];
	}
	//视差计算
	computeDisparity();

	display();
	destroy();
	return 0;
}
