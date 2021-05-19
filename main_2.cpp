#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <math.h>
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
Mat dispImage;				//������ʾ�ĻҶ�ͼ��

unsigned char* initCost;	//��ʼ����
unsigned char* aggrCost_1;
unsigned char* aggrCost_2;
unsigned char* aggrCost_3;
unsigned char* aggrCost_4;
unsigned short* aggrCost;	//�ۺϴ���

float* disparity;	//�Ӳ�
float* disparityRight;
int minDisparity = 0;
int maxDisparity = 64;
int P1 = 10;
int initP2 = 150;

/// <summary>
/// census�任��5x5��
/// </summary>
/// <param name="input">����ͼ������</param>
/// <param name="output">���censusֵ</param>
/// <param name="width">ͼ���</param>
/// <param name="height">ͼ���</param>
void censusTransform(const unsigned char* input, unsigned int* output, int width, int height)
{
	if (input == nullptr || output == nullptr || width <= 5 || height <= 5) {
		printf("Error in censusTransform!\n");
		return;
	}
	unsigned char centerGray = 0;
	//����censusֵ
	for (int i = 2; i < height - 2; i++) {
		for (int j = 2; j < width - 2; j++) {

			//�������صĻҶ�ֵ
			centerGray = input[i * width + j];

			// ����5x5�������������أ�����censusֵ���������ĵ�ļ�Ϊ0��С�ڼ�Ϊ1��
			unsigned int censusValue = 0u;//��25λ��censusֵ
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
/// ����x��y��Hamming����
/// </summary>
/// <param name="x"></param>
/// <param name="y"></param>
/// <returns></returns>
int hammingDistance(unsigned int x, unsigned int y)
{
	unsigned int distance = 0, val = x ^ y;//���
	//ͳ��val�����ƴ���1�ĸ���
	while (val) {
		++distance;
		val &= val - 1;
	}
	return distance;
}
//����ͼƬ
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

	//ͼ���ߡ�������
	image.width = imgLeft.cols;
	image.height = imgLeft.rows;
	int pixelNum = image.width * image.height;

	//���Ҷ���Ϣ���浽image�ṹ����
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


//�����ʼ����
void computeCost() {
	const int width = image.width;
	const int height = image.height;
	const unsigned char* leftGray = image.leftGray;
	const unsigned char* rightGray = image.rightGray;
	const int pixelNum = height * width;	//������

	//census�任
	unsigned int* censusLeft = (unsigned int*)malloc(sizeof(unsigned int) * pixelNum);
	unsigned int* censusRight = (unsigned int*)malloc(sizeof(unsigned int) * pixelNum);
	if (censusLeft == NULL || censusRight == NULL) {
		printf("Error in calculateCost: Malloc Failure\n");
		return;
	}
	censusTransform(leftGray, censusLeft, width, height);
	censusTransform(rightGray, censusRight, width, height);

	//�����ʼ����
	int dispRange = maxDisparity - minDisparity;	//�ӲΧ
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {

			unsigned int censusLeftValue = censusLeft[i * width + j];

			for (int m = minDisparity; m < maxDisparity; ++m) {
				int index = i * width * dispRange + j * dispRange + (m - minDisparity);

				if (j - m < 0 || j - m >= width) {
					initCost[index] = 128;
					continue;
				}

				//�����ʼƥ�����
				const unsigned int censusRightValue = censusRight[i * width + j - m];
				initCost[index] = hammingDistance(censusLeftValue, censusRightValue);
			}

		}
	}

	free(censusLeft);
	free(censusRight);
}

//���۾ۺ�--���ҷ�����
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
				int l2 = d == 0 ? 255 + P1 : aggrCostLastPos[d - 1] + P1;
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

//�Ӳ����--���ӲΧ��ѡ��һ������ֵ��С���Ӳ���Ϊ���ص������Ӳ�
void computeDisparity() {
	int dispRange = maxDisparity - minDisparity;

	const int width = image.width;
	const int height = image.height;
	const unsigned char* leftGray = image.leftGray;
	const unsigned char* rightGray = image.rightGray;
	const int pixelNum = height * width;	//������

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			//�����ӲΧ�ڵ����д���ֵ��ȡ����ֵ��С���Ӳ���Ϊ���ص������Ӳ�
			//�����ӲΧ�ڵ����д���ֵ��ȡ����ֵ��С���Ӳ���Ϊ���ص������Ӳ�
			unsigned short minCost = 0x7fff;
			unsigned short secondMinCost = 0x7fff;
			int bestDisparity = 0;
			for (int m = minDisparity; m < maxDisparity; ++m) {
				int index = dispRange * (i * width + j) + (m - minDisparity);
				if (aggrCost[index] < minCost) {
					minCost = aggrCost[index];
					bestDisparity = m;
				}
			}

			for (int m = minDisparity; m < maxDisparity; m++)
			{
				int index = dispRange * (i * width + j) + (m - minDisparity);
				if (aggrCost[index] < secondMinCost && m != bestDisparity)
					secondMinCost = aggrCost[index];
			}

			if (secondMinCost - minCost <= (unsigned short)(minCost * 0.05))
			{
				disparity[i * width + j] = INVALID_PIXEL;
				continue;
			}
			if (bestDisparity == minDisparity || bestDisparity == maxDisparity)
			{
				disparity[i * width + j] = INVALID_PIXEL;
				continue;
			}
			//ȡ����ֵ��С���Ӳ���Ϊ���ص������Ӳ�
			unsigned short cost_1 = aggrCost[dispRange * (i * width + j) + bestDisparity - 1 - minDisparity];
			unsigned short cost_2 = aggrCost[dispRange * (i * width + j) + bestDisparity + 1 - minDisparity];
			unsigned short denom = max(1, cost_1 + cost_2 - minCost * 2);
			disparity[i * width + j] = bestDisparity + (cost_1 - cost_2) / (2.0 * denom);
		}
	}
}

void computeDisparityRight()
{
	int dispRange = maxDisparity - minDisparity;

	const int width = image.width;
	const int height = image.height;
	const unsigned char* leftGray = image.leftGray;
	const unsigned char* rightGray = image.rightGray;
	const int pixelNum = height * width;	//������

	unsigned short* costLocal = (unsigned short*)malloc(dispRange * sizeof(unsigned short));

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			unsigned short minCost = 0x7fff;
			unsigned short secondMinCost = 0x7fff;
			int bestDisparity = 0;

			for (int d = minDisparity; d < maxDisparity; d++)
			{
				int leftCol = j + d, index = d - minDisparity;
				if (leftCol >= 0 && leftCol < width)
				{
					costLocal[index] = aggrCost[(i * width + leftCol) * dispRange + index];
					if (minCost > costLocal[index])
					{
						minCost = costLocal[index];
						bestDisparity = d;
					}
				}
				else costLocal[index] = 0x7fff;
			}

			for (int d = minDisparity; d < maxDisparity; d++)
			{
				int index = d - minDisparity;
				if (secondMinCost > costLocal[index] && d != bestDisparity)
					secondMinCost = costLocal[index];
			}

			if (secondMinCost - minCost <= minCost * 0.05)
			{
				disparityRight[i * width + j] = INVALID_PIXEL;
				continue;
			}

			if (bestDisparity == minDisparity || bestDisparity == maxDisparity - 1)
			{
				disparityRight[i + width + j] = INVALID_PIXEL;
				continue;
			}

			unsigned short cost_1 = costLocal[bestDisparity - 1 - minDisparity];
			unsigned short cost_2 = costLocal[bestDisparity + 1 - minDisparity];
			unsigned short denom = max(1, cost_1 + cost_2 - 2 * minCost);
			disparityRight[i * width + j] = bestDisparity + (cost_1 - cost_2) / (2.0 * denom);
		}

	free(costLocal);
}

//n*n��ֵ�˲���nΪ������
void medianFilter(unsigned char* grayInput, unsigned char* grayOutput, int width, int height, int n) {
	int maskSize = n * n;
	int* mask = (int*)malloc(sizeof(int) * maskSize);
	if (mask == NULL) {
		printf("Error in medianFilter: Malloc Failure\n");
		return;
	}
	if (n % 2 == 0 || n <= 1) {
		printf("Error in medianFilter: n % 2 == 0 || n <= 1\n");
		return;
	}
	memcpy(grayOutput, grayInput, width * height * sizeof(unsigned char));
	int k = n / 2;
	for (int i = k; i < height - k; ++i) {
		for (int j = k; j < width - k; ++j) {
			int index = 0;
			//ͳ����Ĥ�����е�ֵ
			for (int p = -k; p <= k; ++p) {
				for (int q = -k; q <= k; ++q) {
					int row = i + p;
					int col = j + q;
					mask[index++] = grayInput[row * width + col];
				}
			}
			//����
			std::sort(mask, mask + n * n);
			grayOutput[i * width + j] = mask[maskSize / 2];
		}
	}
	free(mask);
}

void LRCheck()
{
	int dispRange = maxDisparity - minDisparity;

	const int width = image.width;
	const int height = image.height;
	const unsigned char* leftGray = image.leftGray;
	const unsigned char* rightGray = image.rightGray;
	const int pixelNum = height * width;	//������

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{

		}
}

//�����Ӳ����ɻҶ�ͼ
void convertToImage() {
	const int width = image.width;
	const int height = image.height;
	dispImage = cv::Mat(height, width, CV_8UC1);
	int dispRange = maxDisparity - minDisparity;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			const unsigned short disp = disparity[i * width + j];
			if (disparity[i * width + j] == INVALID_PIXEL) {
				dispImage.data[i * width + j] = 0;
			}
			else {
				//��dispӳ�䵽����[0,255]
				dispImage.data[i * width + j] = static_cast<uchar>(((float)disp - minDisparity) / (dispRange) * 255);
			}
		}
	}
}

//��ʾ�Ӳ�ͼ
void display() {
	const int width = image.width;
	const int height = image.height;

	Mat dispMedianImage = cv::Mat(height, width, CV_8UC1);

	medianFilter(dispImage.data, dispMedianImage.data, width, height, 3);
	cv::imshow("�Ӳ�ͼ1", dispImage);
	cv::imshow("�Ӳ�ͼ2", dispMedianImage);
	cv::waitKey(0);
}
//��ʼ��
void init(const char* leftImagePath, const char* rightImagePath) {
	loadImage(leftImagePath, rightImagePath);
	initCost = (unsigned char*)malloc(sizeof(unsigned char) * image.height * image.width * (maxDisparity - minDisparity));
	aggrCost_1 = (unsigned char*)malloc(sizeof(unsigned char) * image.height * image.width * (maxDisparity - minDisparity));
	aggrCost_2 = (unsigned char*)malloc(sizeof(unsigned char) * image.height * image.width * (maxDisparity - minDisparity));
	aggrCost_3 = (unsigned char*)malloc(sizeof(unsigned char) * image.height * image.width * (maxDisparity - minDisparity));
	aggrCost_4 = (unsigned char*)malloc(sizeof(unsigned char) * image.height * image.width * (maxDisparity - minDisparity));
	aggrCost = (unsigned short*)malloc(sizeof(unsigned short) * image.height * image.width * (maxDisparity - minDisparity));
	disparity = (float*)malloc(sizeof(float) * image.height * image.width);
	disparityRight = (float*)malloc(sizeof(float) * image.height * image.width);
}
//�ͷ��ڴ�
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
	free(disparityRight);
}

int main() {
	init("G:\\Other\\image\\left1.png", "G:\\Other\\image\\right1.png");
	//���ۼ���
	computeCost();
	//���۾ۺ�
	costAggregationRow(true);
	costAggregationRow(false);
	costAggregationCol(true);
	costAggregationCol(false);
	int size = (maxDisparity - minDisparity) * image.width * image.height;
	for (int i = 0; i < size; ++i) {
		aggrCost[i] = aggrCost_1[i] + aggrCost_2[i] + aggrCost_3[i] + aggrCost_4[i];
	}
	//�Ӳ����
	computeDisparity();
	computeDisparityRight();
	LRCheck();
	//���ɻҶ�ͼ
	convertToImage();
	display();
	destroy();
	return 0;
}