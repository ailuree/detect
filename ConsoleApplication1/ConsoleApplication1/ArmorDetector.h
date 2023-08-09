#pragma once
#include <iostream>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace cv;
using namespace std;
// װ�װ���ɫ ����곣��
#define RED 1
#define BLUE 0
// �洢װ�װ�����Ľṹ��
struct ArmorParam {
    float Image_bright; // ͼ�񽵵͵�����
    int threshold_value; // threshold��ֵ

    float Light_Area_min; // ��������Сֵ
    float Light_Aspect_ratio; // �����ĳ��������
    float Light_crown; // �������������������

    float Light_angle; // ��������б�Ƕ�
    float Light_Contour_angle; // �����ǶȲ�
    float Light_Contour_Len; // �������Ȳ����

    float Armor_ratio_max; // װ�װ�ĳ����max
    float Armor_ratio_min; // װ�װ�ĳ����min
    float Armor_xDiff; // װ�װ�x�����
    float Armor_yDiff; // װ�װ�y�����
    float Armor_angle_min; // װ�װ�Ƕ�min
    bool Armor_Color; // װ�װ���ɫ
    ArmorParam()//Ĭ�Ϲ��캯��
    {
        Image_bright = -100;
        threshold_value = 25;
        Light_Area_min = 20;
        Light_Aspect_ratio = 0.7;
        Light_crown = 0.5;
        Light_angle = 5;
        Light_Contour_angle = 4.2;
        Light_Contour_Len = 0.25;
        Armor_ratio_max = 5.0;
        Armor_ratio_min = 1.0;
        Armor_xDiff = 0.5;
        Armor_yDiff = 2.0;
        Armor_angle_min = 5;
        Armor_Color = BLUE;
    }
} _Armor;

Mat Adjust_Bright(Mat); // ����ͼ������
Mat Threshold_Demo(Mat); // ��ֵ������
RotatedRect& adjustRec(RotatedRect& rec); // ����rec
bool Found_Contour(Mat, vector<RotatedRect>&); // Ѱ������
bool Identify_board(Mat, vector<RotatedRect>&, vector<RotatedRect>&); // ʶ��װ�װ�
void drawBox(vector<RotatedRect>&, Mat&, Mat&); // ��������
void drawBox1(vector<RotatedRect>&, Mat&, Mat&); //��������