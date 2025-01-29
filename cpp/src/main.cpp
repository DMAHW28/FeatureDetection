//
// Created by Liam Dro on 26/01/2025.
//
#include <tuple>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "corners.hpp"
#include "descriptors.hpp"
using namespace std;

int main() {
    cornersD::Harris harris;
    cornersD::Fast fast;
    string imagePath1 = "/Users/liamdro/PycharmProjects/FeatureDetection/images/P1.JPG";
    string imagePath2 = "/Users/liamdro/PycharmProjects/FeatureDetection/images/P2.JPG";
    // tuple<cv::Mat, int, vector<cv::Point>> H1 = fast.compute_fast_detector(imagePath1);
    // tuple<cv::Mat, int, vector<cv::Point>> H2 = fast.compute_fast_detector(imagePath2);
    tuple<cv::Mat, int, vector<cv::Point>> H1 = harris.compute_harris_detector(imagePath1);
    tuple<cv::Mat, int, vector<cv::Point>> H2 = harris.compute_harris_detector(imagePath2);
    vector<cv::Point> pts1 = get<2>(H1);
    vector<cv::Point> pts2 = get<2>(H2);
    tuple<vector<cv::Point>, vector<cv::Point>, vector<cv::Point>> Matches = desc::compute_matching_points(imagePath1, imagePath2, pts1, pts2, 33,  "euclidean" , "ratio" ,  10);
    cv::Mat matching_img = desc::display_matching(get<0>(H1), get<0>(H2), get<1>(Matches), get<2>(Matches), get<0>(Matches), 30);

    for (int i=0; i<3; ++i){
        if (i == 0){
            string title = "Number of corners points detected : " + to_string(get<1>(H1));
            cv::imshow(title, get<0>(H1));
        }
        else if (i == 1){
            string title = "Number of corners points detected : " + to_string(get<1>(H2));
            cv::imshow(title, get<0>(H2));
        }
        else{
            cv::imshow("Descriptors", matching_img);
        }
        cv::waitKey(0);
    }
    return 0;
}
