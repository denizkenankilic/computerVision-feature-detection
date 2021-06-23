#include <QCoreApplication>
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "vector"
#include <QDebug>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    Mat imageSiftFeature, imageSurfFeature,imagefastfeature,imageOrbFeature,imageMserFeature;


    Mat image1= imread(".../55.jpg", CV_LOAD_IMAGE_COLOR); // Edit the path according to your folders
     imagefastfeature = image1.clone();
     imageSurfFeature = image1.clone();
     imageSiftFeature = image1.clone();
     imageOrbFeature = image1.clone();
     imageMserFeature = image1.clone();
    imshow("image1", image1);


    vector< KeyPoint > keypointsFast;


    FastFeatureDetector fast(10); // threshold for detection

    fast.detect(image1,keypointsFast);
    drawKeypoints(image1,   // original image
    keypointsFast,              // vector of keypoints
    imagefastfeature,       // the output image
    Scalar(255,255,0),    // keypoint color
    DrawMatchesFlags::DRAW_OVER_OUTIMG); //drawing flag

    imshow(" Fast feature detection ", imagefastfeature);

    vector< KeyPoint > keypointsSift;

    SiftFeatureDetector sift (5000.); // threshold

    sift.detect(image1,keypointsSift);


    cv::drawKeypoints(image1,// original image
    keypointsSift,// vector of keypoints
    imageSiftFeature,// the resulting image
    cv::Scalar(255,255,0),// color of the points
    cv::DrawMatchesFlags::DRAW_OVER_OUTIMG); //flag

    imshow("Sift feature detection ", imageSiftFeature);

    cout << keypointsFast.size() << " Fast " << endl;
    cout << keypointsSift.size() << " Sift " << endl;


    keypointsFast.at(5).pt.x = 5;


    vector< KeyPoint > keypointsSurf;

    SurfFeatureDetector surf (10.); // threshold

    surf.detect(image1,keypointsSurf);

    cv::drawKeypoints(image1,// original image
    keypointsSurf,// vector of keypoints
    imageSurfFeature,// the resulting image
    cv::Scalar(255,255,0),// color of the points
    cv::DrawMatchesFlags::DRAW_OVER_OUTIMG); //flag

    imshow("Surf feature detection ", imageSurfFeature);

    cout << keypointsSurf.size() << " Surf "<< endl;

    vector< KeyPoint > keypointsORB;


    OrbFeatureDetector detector;

    detector.detect(image1,keypointsORB);

    cv::drawKeypoints(image1,// original image
    keypointsORB,// vector of keypoints
    imageOrbFeature,// the resulting image
    cv::Scalar(255,255,0),// color of the points
    cv::DrawMatchesFlags::DRAW_OVER_OUTIMG); //flag

    imshow("ORB feature detection ", imageOrbFeature);

    cout << keypointsORB.size() << " Orb " << endl;


     vector< KeyPoint > keypointsmser;

    MserFeatureDetector mser;

    mser.detect(image1,keypointsmser);
    cv::drawKeypoints(image1,// original image
    keypointsmser,// vector of keypoints
    imageMserFeature,// the resulting image
    cv::Scalar(255,255,0),// color of the points
    cv::DrawMatchesFlags::DRAW_OVER_OUTIMG); //flag


    imshow("Mser feature detection ", imageMserFeature);

     cout << keypointsmser.size() << " Mser " << endl;


    return a.exec();
}
