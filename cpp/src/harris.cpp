//
// Created by Liam Dro on 26/01/2025.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>
#include <stdexcept>
#include "corners.hpp"
using namespace std;
using namespace cornersD;
cv::Mat Harris::harris_detector_score(const string& image_file, const string& window_type, int window_size, float k, float sigma) {
    /*
    Compute the Harris corner score for the image.

    Parameters:
    - image_file: Path to the image file.
    - window_type: Type of the filter (e.g., "Gaussian" or "Rectangular").
    - window_size: Size of the filter kernel.
    - k: Harris criterion constant.
    - sigma: Standard deviation of the Gaussian filter.

    Returns:
    - C: Harris score matrix.
    */

    // Read the image in grayscale
    cv::Mat image = cv::imread(image_file, cv::IMREAD_GRAYSCALE);

    // Check if the image was loaded successfully
    if (image.empty()) {
        const string error_msg = "Error: Could not open or find the image " + image_file;
        throw invalid_argument(error_msg);
    }

    cv::Size kernel_size = cv::Size(window_size, window_size);

    // Compute the gradients (Ix, Iy)
    pair<cv::Mat, cv::Mat> gradients = compute_gradient(image);
    cv::Mat Ix = gradients.first;
    cv::Mat Iy = gradients.second;

    // Compute the products of gradients
    cv::Mat Ixx = Ix.mul(Ix);
    cv::Mat Iyy = Iy.mul(Iy);
    cv::Mat Ixy = Ix.mul(Iy);

    // Apply the chosen window type for filtering
    cv::Mat Sxx, Syy, Sxy;

    if (window_type == "Rectangular") {
        tuple<cv::Mat, cv::Mat, cv::Mat> F = rectangularFilter(Ixx, Iyy, Ixy, kernel_size);
        Sxx = get<0>(F); Syy = get<1>(F); Sxy = get<2>(F);
    }
    else if (window_type == "Gaussian") {
        tuple<cv::Mat, cv::Mat, cv::Mat> F = gaussianFilter(Ixx, Iyy, Ixy, kernel_size, sigma);
        Sxx = get<0>(F); Syy = get<1>(F); Sxy = get<2>(F);
    }
    else {
        throw invalid_argument("Invalid window type: Choose 'Rectangular' or 'Gaussian'.");
    }

    // Compute the Harris corner response matrix
    cv::Mat det_M = Sxx.mul(Syy) - Sxy.mul(Sxy);
    cv::Mat trace_M = Sxx + Syy;
    cv::Mat C = det_M - k * trace_M.mul(trace_M);
    return C;
}
tuple<cv::Mat, int, vector<cv::Point>> Harris::harris_detector_image(const string& image_file, const cv::Mat& corners){
    /*
    Display detected corners using the Harris detector on the input image.

    Parameters:
    - image_file: Path to the image file (the image to process).
    - corners: Harris detected corners (binary matrix with 1.0 for corners).

    Returns:
    - harris_image: Image with detected corners marked.
    - corner_number: Number of detected corners.
    - points: List of detected corner points.
    */

    //  Read the image
    cv::Mat harris_image = cv::imread(image_file);

    // Apply detected points on image and count the number of points detected
    cv::Scalar color(0, 0, 255);
    int corner_number = 0;
    vector<cv::Point> points;
    for (int i = 0; i < harris_image.rows; ++i) {
        for (int j = 0; j < harris_image.cols; ++j){
            if (corners.at<float>(i, j) == 1.0){
                cv::Point p(j, i);
                points.push_back(p);
                ++corner_number;
                // Draw red circles on detected corners
                cv::circle(harris_image, p, 2, color, -1);
            }
        }
    }
    return make_tuple(harris_image, corner_number, points);
}
tuple<cv::Mat, int, vector<cv::Point>> Harris::compute_harris_detector(const string& image_file, const string& window_type, int window_size, float k, float sigma, bool nms, int nms_window, float threshold){
    /*
    Computes the Harris corner detector with optional non-maxima suppression.

    Parameters:
    - image_file: Path to the image file (the image to process).
    - window_type: Type of the filter (e.g., "Gaussian" or "Rectangular").
    - window_size: Size of the filter kernel.
    - k: Harris criterion constant.
    - nms: Boolean flag to apply non-maxima suppression.
    - sigma: Standard deviation of the Gaussian filter.
    - threshold: Threshold for corner detection based on Harris score.
    - nms_window: Size of the window for non-maxima suppression.

    Returns:
    - harris_image: Image with detected corners marked.
    - corner_number: Number of detected corners.
    - points: List of detected corner points.
    */

    // Compute the Harris corner score.
    cv::Mat score = harris_detector_score(image_file, window_type, window_size, k, sigma);

    // Apply nms if requested
    if (nms){
        score = non_maxima_suppression(score, nms_window);
    }
    // Estimate detection threshold based on the maximum score
    double max_val;
    cv::minMaxLoc(score, nullptr, &max_val);
    float rate = max_val * threshold;
    // Estimate corners by applying the threshold
    cv::Mat corners = cv::Mat::zeros(score.rows, score.cols, CV_32F);
    for (int i=0; i<score.rows; ++i){
        for (int j = 0; j < score.cols; ++j) {
            if (score.at<float>(i, j) > rate){
                corners.at<float>(i, j) = 1.0;
            }
        }
    }
    return harris_detector_image(image_file,  corners);
}

