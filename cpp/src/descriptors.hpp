//
// Created by Liam Dro on 26/01/2025.
//
#ifndef FEATUREDETECTION_DESCRIPTORS_H
#define FEATUREDETECTION_DESCRIPTORS_H
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <tuple>

using namespace std;
namespace desc{
    tuple<cv::Mat, vector<cv::Point>> extract_descriptors_basic(const cv::Mat& image, const vector<cv::Point>& points, int window_size = 3);
    float distance_metric_euclidean(const cv::Mat& desc1, const cv::Mat& desc2);
    float distance_metric_correlation(const cv::Mat& desc1, const cv::Mat& desc2);
    float distance_metric_chi_square(const cv::Mat& desc1, const cv::Mat& desc2);
    cv::Mat compute_distance(const cv::Mat& desc1, const cv::Mat& desc2, const string& metric = "euclidean");
    tuple<vector<cv::Point>, vector<cv::Point>, vector<cv::Point>> compute_matching_points(const string& img1_file, const string& img2_file, const vector<cv::Point>& pts1, const vector<cv::Point>& pts2, int windows_size = 3, const string& metric = "euclidean", const string& match = "ratio", float ratio_threshold=0.8, const string& desc_type = "basic", int radius = 1, int n_points = 8);
    vector<cv::Point> matching_points_ratio(const cv::Mat& desc1, const cv::Mat& desc2, const string& metric = "euclidean", float ratio_threshold = 0.8);
    cv::Mat arg_sort(const cv::Mat& distance_col);
    cv::Mat display_matching(const cv::Mat& img1, const cv::Mat& img2, const vector<cv::Point>& pts1, const vector<cv::Point>& pts2, const vector<cv::Point>& matches, int nbp);
    int compute_lbp(const cv::Mat& image, int x, int y, int radius = 1, int n_points = 8);
    tuple<cv::Mat, vector<cv::Point>> extract_descriptors_lbp(const cv::Mat& image, const vector<cv::Point>& points, int radius = 1, int n_points = 8);
    cv::Mat histogram(const cv::Mat& x, int bins = 256, int range_max = 256);
}
#endif //FEATUREDETECTION_DESCRIPTORS_H