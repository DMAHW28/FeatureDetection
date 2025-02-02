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
    tuple<cv::Mat, int, vector<cv::Point>> H1 = fast.compute_fast_detector(imagePath1, 12, 20, true, 7);
    tuple<cv::Mat, int, vector<cv::Point>> H2 = fast.compute_fast_detector(imagePath2, 12, 20, true, 7);
    // tuple<cv::Mat, int, vector<cv::Point>> H1 = harris.compute_harris_detector(imagePath1, "Gaussian", 3, 0.055, 1, true, 5, 0.01);
    // tuple<cv::Mat, int, vector<cv::Point>> H2 = harris.compute_harris_detector(imagePath2, "Gaussian", 3, 0.055, 1, true, 5, 0.01);
    vector<cv::Point> pts1 = get<2>(H1);
    vector<cv::Point> pts2 = get<2>(H2);
    // euclidean = 0.4; chi = 0.1 => basic
    tuple<vector<cv::Point>, vector<cv::Point>, vector<cv::Point>> Matches = desc::compute_matching_points(imagePath1, imagePath2, pts1, pts2, 11, "chi", "ratio", 0.5, "lbp", 3, 12);
    cv::Mat matching_img = desc::display_matching(get<0>(H1), get<0>(H2), get<1>(Matches), get<2>(Matches), get<0>(Matches), 50);
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
            string title = "Number of matching points detected : " + to_string(get<0>(Matches).size());
            cv::imshow(title, matching_img);

        }
        cv::waitKey(0);
    }
    cv::imwrite("/Users/liamdro/PycharmProjects/FeatureDetection/docs/fast_1_cpp.png", get<0>(H1));
    cv::imwrite("/Users/liamdro/PycharmProjects/FeatureDetection/docs/fast_2_cpp.png", get<0>(H2));
    cv::imwrite("/Users/liamdro/PycharmProjects/FeatureDetection/docs/matching_points_fast_cpp.png", matching_img);
    return 0;
}
