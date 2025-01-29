//
// Created by Liam Dro on 26/01/2025.
//
#ifndef FEATUREDETECTION_CORNERS_H
#define FEATUREDETECTION_CORNERS_H
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <tuple>
using namespace std;
namespace cornersD {
    inline cv::Mat rotate_image(const cv::Mat& image, float angle = 30){
        /*
        Rotates the input image by a specified angle.

        Parameters:
        - image: Input image to be rotated.
        - angle: Rotation angle in degrees (default is 30).

        Returns:
        - rotated: The rotated image.
        */
        // Output image and size of the image
        cv::Mat rotated;
        int h = image.rows;
        int w = image.cols;
        // Define the center of rotation, get the rotation matrix, and apply the rotation
        cv::Point2f center(w / 2.0f, h / 2.0f);
        cv::Mat M = cv::getRotationMatrix2D(center, angle, 1.0);
        cv::warpAffine(image, rotated, M, cv::Size(w, h));
        return rotated;
    }

    inline cv::Mat non_maxima_suppression(const cv::Mat& score, int window_size = 7){
        /*
        Apply non-maxima suppression to the given score matrix.

        Parameters:
        - score: Harris score matrix.
        - window_size: Size of the sliding window for local maxima suppression.

        Returns:
        - local_max: Harris score matrix after non-maxima suppression, where non-maxima values are set to zero.
        */
        // Copy the score matrix to store the result
        cv::Mat local_max = score.clone();
        int half_window = window_size / 2;
        for (int i = half_window; i < score.rows - half_window; ++i) {
            for (int j = half_window; j < score.cols - half_window; ++j) {
                // Define a window centered at (i, j)
                cv::Rect roi(j - half_window, i - half_window, window_size, window_size);
                cv::Mat window = score(roi);
                double max_val;
                // Find the maximum value in the window
                cv::minMaxLoc(window, nullptr, &max_val);
                // Set pixels that are not local maxima to zero
                if (score.at<float>(i, j) != max_val) {
                    local_max.at<float>(i, j) = 0;
                }
            }
        }
        return local_max;
    }

    inline pair<cv::Mat, cv::Mat> compute_gradient(cv::Mat& image){
        /*
        Compute image gradients using finite differences along the x and y axes.

        Parameters:
        - image: Input image.

        Returns:
        - Ix: Gradient along the x-axis.
        - Iy: Gradient along the y-axis.
        */
        // Size of image
        int h = image.rows;
        int w = image.cols;
        cv::Mat Ix = cv::Mat::zeros(h, w, CV_32F);
        cv::Mat Iy = cv::Mat::zeros(h, w, CV_32F);
        // Compute gradient along x-axis and y-axis using finite differences
        for (int y = 1; y < h - 1; ++y) {
            for (int x = 1; x < w - 1; ++x) {
                Ix.at<float>(y, x) = (image.at<uchar>(y, x + 1) - image.at<uchar>(y, x - 1)) / 2.0f;
                Iy.at<float>(y, x) = (image.at<uchar>(y + 1, x) - image.at<uchar>(y - 1, x)) / 2.0f;
            }
        }
        return make_pair(Ix, Iy);
    }

    inline tuple<cv::Mat, cv::Mat, cv::Mat> rectangularFilter(cv::Mat& Gx, cv::Mat& Gy, cv::Mat& Gxy, cv::Size kernel_size){
        /*
        Apply a rectangular filter to the image gradients.

        Parameters:
        - Gx: Image gradient along the x-axis.
        - Gy: Image gradient along the y-axis.
        - Gxy: Image gradient in both x and y directions.
        - kernel_size: Size of the rectangular filter kernel.

        Returns:
        - Sxx: Filtered image gradient along the x-axis.
        - Syy: Filtered image gradient along the y-axis.
        - Sxy: Filtered image gradient in both x and y directions.
        */
        cv::Mat Sxx, Syy, Sxy;
        cv::boxFilter(Gx, Sxx, -1, kernel_size);
        cv::boxFilter(Gy, Syy, -1, kernel_size);
        cv::boxFilter(Gxy, Sxy, -1, kernel_size);
        return make_tuple(Sxx, Syy, Sxy);
    }

    inline tuple<cv::Mat, cv::Mat, cv::Mat> gaussianFilter(cv::Mat& Gx, cv::Mat& Gy, cv::Mat& Gxy, cv::Size kernel_size, float sigma = 1){
        /*
        Apply a Gaussian filter to the image gradients.

        Parameters:
        - Gx: Image gradient along the x-axis.
        - Gy: Image gradient along the y-axis.
        - Gxy: Image gradient in both x and y directions.
        - kernel_size: Size of the Gaussian filter kernel.
        - sigma: Standard deviation of the Gaussian filter.

        Returns:
        - Sxx: Filtered image gradient along the x-axis.
        - Syy: Filtered image gradient along the y-axis.
        - Sxy: Filtered image gradient in both x and y directions.
        */
        cv::Mat Sxx, Syy, Sxy;
        cv::GaussianBlur(Gx, Sxx, kernel_size, sigma);
        cv::GaussianBlur(Gy, Syy, kernel_size, sigma);
        cv::GaussianBlur(Gxy, Sxy, kernel_size, sigma);
        return make_tuple(Sxx, Syy, Sxy);
    }

    class Harris {
    public:
        /*
        Harris Corner Detector class

        Methods:
        - harris_detector_score: Compute the Harris corner score for the image.
        - harris_detector_image: Display detected corners using the Harris detector on the input image.
        - compute_harris_detector: Computes the Harris corner detector with optional non-maxima suppression.
        */
        cv::Mat harris_detector_score(const string& image_file, const string& window_type="Gaussian", int window_size=3, float k=0.055, float sigma=1, bool rotate=false, float angle=30);
        tuple<cv::Mat, int, vector<cv::Point>> harris_detector_image(const string& image_file, const cv::Mat& corners, bool rotate = false, float angle = 30.0, float r = 2);
        tuple<cv::Mat, int, vector<cv::Point>> compute_harris_detector(const string& image_file, const string& window_type="Gaussian", int window_size=3, float k=0.055, float sigma = 1, bool nms = true, int nms_window = 5, bool rotate = false, float angle = 30, float r = 2, float threshold = 0.01);
    };

    class Fast {
    public:
        /*
        FAST Corner Detector class

        Methods:
        - fast_detector_score: Computes the FAST corner score for the image.
        - fast_detector_image: Display detected corners using the FAST detector on the input image.
        - compute_fast_detector: Computes the FAST corner detector with optional parameters for the threshold and rotation.
        - threshold_ver: Verifies the threshold condition for the FAST detector's corner detection.
        */
        cv::Mat fast_detector_score(const string& image_file, int n=12, int threshold = 20, bool rotate=false, float angle=30);
        tuple<cv::Mat, int, vector<cv::Point>> fast_detector_image(const string& image_file, const cv::Mat& corners, bool rotate = false, float angle = 30.0, float r = 2);
        tuple<cv::Mat, int, vector<cv::Point>> compute_fast_detector(const string& image_file, int n=12, int threshold = 20, bool nms = true, int nms_window = 5, bool rotate = false, float angle = 30, float r = 2);
        tuple<bool, float> threshold_ver(const vector<uchar>& circle, float Ip, int threshold = 20, int n = 12);
    };
};
#endif //FEATUREDETECTION_CORNERS_H
