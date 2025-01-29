//
// Created by Liam Dro on 26/01/2025.
//
#include <opencv2/opencv.hpp>
#include <iostream>
#include "corners.hpp"
#include <vector>
#include <tuple>
using namespace std;
using namespace cornersD;

cv::Mat Fast::fast_detector_score(const string& image_file, int n, int threshold, bool rotate, float angle){
    /*
    Compute the Fast corner score for the image.

    Parameters:
    - image_file: Path to the image file.
    - n: Number of contiguous pixels.
    - threshold: threshold value.
    - rotate: Boolean flag to indicate if the image should be rotated.
    - angle: Rotation angle in degrees.

    Returns:
    - C: Fast score matrix.
    */

    // Read the image
    cv::Mat image = cv::imread(image_file, cv::IMREAD_GRAYSCALE);

    // Check if the image was loaded successfully
    if (image.empty()) {
        throw invalid_argument("Error: Could not open or find the image " + image_file);
    }

    // Rotate the image if requested
    if (rotate) {
        image = rotate_image(image, angle);
    }
    // List of positions of pixels in circle of radius 3 around current pixels
    vector<pair<int, int>> circle_pixels = {
            {0, 3}, {1, 3}, {2, 2}, {3, 1},
            {3, 0}, {3, -1}, {2, -2}, {1, -3},
            {0, -3}, {-1, -3}, {-2, -2}, {-3, -1},
            {-3, 0}, {-3, 1}, {-2, 2}, {-1, 3}
    };

    // Image size
    int h = image.rows;
    int w = image.cols;
    cv::Mat C = cv::Mat::zeros(image.size(), CV_32F);
    for (int y = 3; y< h-3; ++y){
        for (int x = 3; x < w-3; ++x) {
            uchar Ip = image.at<uchar>(y, x);
            // Intensities of 16 pixels in the circle
            vector<uchar> circle_intensities;
            for (const auto& pixel : circle_pixels) {
                int new_y = y + pixel.first;
                int new_x = x + pixel.second;
                if (new_y >= 0 && new_y < image.rows && new_x >= 0 && new_x < image.cols) {
                    circle_intensities.push_back(image.at<uchar>(new_y, new_x));
                }
            }
            // Test if pixel is a corner using threshold_ver and add pixel score
            tuple<bool, float> thresholdVer_r = threshold_ver(circle_intensities, Ip, n);
            if (get<0>(thresholdVer_r)){
                C.at<float>(y, x) = get<1>(thresholdVer_r);
            }
        }
    }
    return C;
}

tuple<cv::Mat, int, vector<cv::Point>> Fast::fast_detector_image(const string& image_file, const cv::Mat& corners, bool rotate, float angle, float r){
    /*
    Display detected corners using the FAST detector on the input image.

    Parameters:
    - image_file: Path to the image file (the image to process).
    - corners: Harris detected corners (binary matrix with 1.0 for corners).
    - r: Radius of the circles drawn around detected corners.
    - rotate: Boolean flag to indicate if the image should be rotated.
    - angle: Rotation angle in degrees.

    Returns:
    - harris_image: Image with detected corners marked.
    - corner_number: Number of detected corners.
    - points: List of detected corner points.
    */
    //  Read the image
    cv::Mat harris_image = cv::imread(image_file);
    //  Rotate the image if requested
    if (rotate){
        harris_image = rotate_image(harris_image, angle);
    }

    // Apply detected points on image and count the number of points detected
    cv::Scalar color(0, 0, 255);
    int corner_number = 0;
    vector<cv::Point> points;
    for (int i = 0; i < harris_image.rows; ++i) {
        for (int j = 0; j < harris_image.cols; ++j){
            if (corners.at<float>(i, j) > 0.0){
                cv::Point p(j, i);
                points.push_back(p);
                ++corner_number;
                cv::circle(harris_image, p, r, color, -1);
            }
        }
    }
    return make_tuple(harris_image, corner_number, points);
}
tuple<cv::Mat, int, vector<cv::Point>> Fast::compute_fast_detector(const string& image_file, int n, int threshold, bool nms, int nms_window, bool rotate, float angle, float r){
    /*
    Computes the FAST corner detector with optional non-maxima suppression.

    Parameters:
    - image_file: Path to the image file (the image to process).
    - n: Number of contiguous pixels.
    - threshold: threshold value.
    - nms: Boolean flag to apply non-maxima suppression.
    - nms_window: Size of the window for non-maxima suppression.
    - rotate: Boolean flag to indicate if the image should be rotated.
    - angle: Rotation angle in degrees.
    - r: Radius of the circles drawn around detected corners.

    Returns:
    - harris_image: Image with detected corners marked.
    - corner_number: Number of detected corners.
    - points: List of detected corner points.
    */
    // Compute the Fast corner score.
    cv::Mat score = fast_detector_score(image_file, n, threshold, rotate, angle);
    // Apply nms if requested
    if (nms){
        score = non_maxima_suppression(score, nms_window);
    }

    // Estimate corners
    cv::Mat corners = cv::Mat::zeros(score.rows, score.cols, CV_32F);
    for (int i=0; i<score.rows; ++i){
        for (int j = 0; j < score.cols; ++j) {
            if (score.at<float>(i, j) > 0.0){
                corners.at<float>(i, j) = 1.0;
            }
        }
    }
    return fast_detector_image(image_file,  corners, rotate, angle, r);
}

tuple<bool, float> Fast::threshold_ver(const vector<uchar>& circle, float Ip, int threshold, int n) {
    /*
    Verifies the threshold condition for the FAST detector's corner detection

    Parameters:
    - circle: // Intensities of 16 pixels around current pixel.
    - Ip: Intensity of current pixel.
    - threshold: threshold value.
    - n: Number of contiguous pixels.

    Returns:
    - bool: true if pixel is a corner else false
    - score: Fast score for current pixel.
    */
    // Define High and Low threshold
    int threshold_low = Ip - threshold;
    int threshold_high = Ip + threshold;

    cv::Mat circle_mat(circle, true);

    // Define Low and High pixels than the threshold
    cv::Mat_<float> high = (circle_mat > threshold_high) / 255;
    cv::Mat_<float> low = (circle_mat < threshold_low) / 255;
    // Define a binary vector that indicate if pixel is more high or more low than the threshold
    cv::Mat_<float> binary_mat = high | low;

    // For circularity
    cv::Mat_<float> bv;
    cv::hconcat(binary_mat.t(), binary_mat.t(), bv);
    // Convolution with a one values filter
    cv::Mat filter_n = cv::Mat::ones(1, n, CV_32F);

    cv::Mat conv_result;
    cv::filter2D(bv, conv_result, -1, filter_n, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);

    // Estimate corners based on the convolution maximum value
    double max_conv;
    cv::minMaxLoc(conv_result, nullptr, &max_conv);

    if (max_conv >= n) {
        // Estimate the score
        cv::Mat_<float> diff;
        cv::absdiff(circle_mat, cv::Scalar(Ip), diff);
        float score = (float) cv::sum(diff)[0];
        return make_tuple(true, score);
    } else {
        return make_tuple(false, 0.0f);
    }
}