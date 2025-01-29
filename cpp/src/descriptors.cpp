//
// Created by Liam Dro on 26/01/2025.
//
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <tuple>
#include "descriptors.hpp"
#include <algorithm>
#include <numeric>
#include <random>
using namespace std;

tuple<cv::Mat, vector<cv::Point>> desc::extract_descriptors(const cv::Mat& image, const vector<cv::Point>& points, int window_size){
    /*
    Extract intensities block around key points

    Parameters:
    - image: The image to process.
    - points: List of detected corner points.
    - window_size: Size of the window.

    Returns:
    - descriptor_mat: Intensities block matrix around key point.
    - desc_points: List of detected corner points that are extracted.
    */
    vector<cv::Point> desc_points;
    vector<cv::Mat> descriptor;
    int half_window = window_size/2;
    for (const auto& pixel: points){
        int x = pixel.x;
        int y = pixel.y;
        if (!(y - half_window < 0 || y + half_window >= image.rows || x - half_window < 0 || x + half_window >= image.cols)){
            // Define neighbor window
            cv::Rect roi(x - half_window, y - half_window, window_size, window_size);
            cv::Mat window = image(roi);
            if (!window.isContinuous()) {
                window = window.clone();
            }
            // Vectorised and add the descriptor
            cv::Mat flat = window.reshape(1, 1);
            descriptor.push_back(flat);
            desc_points.push_back(pixel);
        }
    }
    // Convert descriptor list to descriptor matrix
    cv::Mat descriptor_mat;
    if (!descriptor.empty()) {
        cv::vconcat(descriptor, descriptor_mat);
    }
    else {
        cerr << "Empty Matrix ! " << endl;
        descriptor_mat = cv::Mat();
    }
    return make_tuple(descriptor_mat, desc_points);
}
float desc::distance_metric_euclidean(const cv::Mat& desc1, const cv::Mat& desc2){
    /*
    Compute Euclidean distance between two descriptors

    Parameters:
    - desc1: descriptor vector of image1.
    - desc2: descriptor vector of image2.

    Returns:
    - float: Euclidean distance between two descriptors.
    */
    cv::Mat_<float> diff = (desc1 - desc2);
    float score = (float) cv::sum(diff.mul(diff))[0];
    return sqrt(score);
}
float desc::distance_metric_correlation(const cv::Mat& desc1, const cv::Mat& desc2){
    /*
    Compute Correlation distance between two descriptors

    Parameters:
    - desc1: descriptor vector of image1.
    - desc2: descriptor vector of image2.

    Returns:
    - float: Correlation distance between two descriptors.
    */
    cv::Mat_<float> desc11 = desc1.mul(desc1);
    cv::Mat_<float> desc22 = desc2.mul(desc2);
    cv::Mat_<float> desc12 = desc1.mul(desc2);
    float score_1 = (float) cv::sum(desc11)[0];
    float score_2 = (float) cv::sum(desc22)[0];
    float correlation = (float) cv::sum(desc12)[0];
    correlation = correlation / (sqrt(score_1) * sqrt(score_2));
    return correlation;
}
cv::Mat desc::compute_distance(const cv::Mat& desc1, const cv::Mat& desc2, const string& metric){
    /*
    Compute distances between the descriptors of the 2 images

    Parameters:
    - desc1: descriptors matrix of image1.
    - desc2: descriptors matrix of image2.
    - metric: metric for compute distance (eg: Euclidean, Correlation, etc.).
    Returns:
    - distances: Distance between the descriptors of the 2 images.
    */
    cv::Mat distances = cv::Mat::zeros(desc2.rows, desc1.rows, CV_32F);
    // For each descriptor of image 1 we compute distance with a metric
    for (int i = 0; i < distances.cols; ++i) {
        // Define descriptor of image 1
        cv::Mat window_1 = desc1.row(i);
        for (int j = 0; j < distances.rows; ++j) {
            // Define descriptor of image 2
            cv::Mat window_2 = desc2.row(j);

            // Compute distance between 2 descriptors with a metric
            if (metric == "euclidean"){
                distances.at<float>(j, i) = desc::distance_metric_euclidean(window_1, window_2);
            }
            else if (metric == "correlation"){
                distances.at<float>(j, i) = desc::distance_metric_correlation(window_1, window_2);
            }
            else{
                throw invalid_argument("Error: Metric doesn't exist not metric: " + metric);
            }
        }
    }
    return distances;
}
tuple<vector<cv::Point>, vector<cv::Point>, vector<cv::Point>> desc::compute_matching_points(const string& img1_file, const string& img2_file, const vector<cv::Point>& pts1, const vector<cv::Point>& pts2, int windows_size , const string& metric , const string& match , float ratio_threshold){
    /*
    Compute matching between keypoints

    Parameters:
    - desc1: descriptors matrix of image1.
    - desc2: descriptors matrix of image2.
    - metric: metric for compute distance (eg: Euclidean, Correlation, etc.).
    Returns:
    - distances: Distance between the descriptors of the 2 images.
    */
    cv::Mat img1 = cv::imread(img1_file, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(img2_file, cv::IMREAD_GRAYSCALE);
    vector<cv::Point> matches;
    tuple<cv::Mat, vector<cv::Point>> extractor1 = extract_descriptors(img1, pts1, windows_size);
    tuple<cv::Mat, vector<cv::Point>> extractor2 = extract_descriptors(img2, pts2, windows_size);
    if (match == "ratio"){
        matches = desc::matching_points_ratio(get<0>(extractor1), get<0>(extractor2), metric, ratio_threshold);
    }
    else{
        throw invalid_argument("Error: Match don't exist not match: " + match);
    }
    return make_tuple(matches, get<1>(extractor1), get<1>(extractor2));
}
vector<cv::Point> desc::matching_points_ratio(const cv::Mat& desc1, const cv::Mat& desc2, const string& metric, float ratio_threshold){
    cv::Mat distance12 = desc::compute_distance(desc1, desc2, metric);
    vector<cv::Point> matches;
    for (int i = 0; i < distance12.cols; ++i){
        cv::Mat sorted_indices = desc::arg_sort(distance12.col(i));
        float d1 = distance12.at<float>(sorted_indices.at<int>(0, 0), i);
        float d2 = distance12.at<float>(sorted_indices.at<int>(0, 1), i);
        if (d1 != 0 && d2 != 0){
            if (d1/d2 < ratio_threshold){
                cv::Point p(i, sorted_indices.at<int>(0, 0));
                matches.push_back(p);
            }
        }
    }
    return matches;
}
cv::Mat desc::arg_sort(const cv::Mat& distance_col){
    vector<float> values(distance_col.begin<float>(), distance_col.end<float>());
    vector<int> indices(values.size());
    iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(), [&values](int i1, int i2) {
        return values[i1] < values[i2];
    });
    cv::Mat sorted_indices(indices, true);
    return sorted_indices.reshape(1, 1);
}
//cv::Mat desc::display_matching(string img1_file, string img2_file, vector<cv::Point> pts1, vector<cv::Point> pts2, vector<cv::Point> matches, int nbp){
cv::Mat desc::display_matching(const cv::Mat& img1, const cv::Mat& img2, const vector<cv::Point>& pts1, const vector<cv::Point>& pts2, const vector<cv::Point>& matches, int nbp){
    //cv::Mat img1 = cv::imread(img1_file);
    //cv::Mat img2 = cv::imread(img2_file);
    if (img1.empty() || img2.empty()) {
        cerr << "Error: Could not open or find the image!" << endl;
        return cv::Mat();
    }
    int h1 = img1.rows, h2 = img2.rows;
    int w1 = img1.cols, w2 = img2.cols;
    cv::Mat canvas = cv::Mat::zeros(max(h1, h2), w1 + w2, img1.type());
    img1.copyTo(canvas(cv::Rect(0, 0, w1, h1)));
    img2.copyTo(canvas(cv::Rect(w1, 0, w2, h2)));
    random_device rd;
    mt19937 rng(rd());
    uniform_int_distribution<int> uni(0, 255);

    for (int k = 0; k < min(nbp, static_cast<int>(matches.size())); ++k) {
        cv::Point idx = matches[k];
        if (idx.x >= pts1.size() || idx.y >= pts2.size()) {
            cerr << "Index Out of range !" << endl;
            continue;
        }
        cv::Point p1 = pts1[idx.x];
        cv::Point p2 = pts2[idx.y];
        cv::Scalar color(uni(rng), uni(rng), uni(rng));
        p2.x += w1;
        cv::circle(canvas, p1, 5, color, -1);
        cv::circle(canvas, p2, 5, color, -1);
        cv::line(canvas, p1, p2, color, 1);
    }
    return canvas;
}