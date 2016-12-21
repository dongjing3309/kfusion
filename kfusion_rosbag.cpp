/*
Copyright (c) 2011-2013 Gerhard Reitmayr, TU Graz

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

// read rosbag file as input

#include "kfusion.h"
#include "helpers.h"
//#include "interface.h"
#include "perfstats.h"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstring>

#include <GL/glut.h>

#include <rosbag/view.h>
#include <rosbag/bag.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cvwimage.h>


#include <cv_bridge/cv_bridge.h>
#include <compressed_depth_image_transport/compression_common.h>

using namespace std;
using namespace TooN;


KFusion kfusion;
Image<uchar4, HostDevice> lightScene, trackModel, lightModel, texModel;

// no need for double buffer for offline mode
Image<uint16_t, HostDevice> depthImage;
Image<uchar3, HostDevice> rgbImage;

const float3 light = make_float3(1, 1, -1.0);
const float3 ambient = make_float3(0.1, 0.1, 0.1);

SE3<float> initPose;

int counter = 0;
int integration_rate = 2;
bool reset = true;
bool should_integrate = true;
bool render_texture = false;

Image<float3, Device> pos, normals;
Image<float, Device> dep;

SE3<float> preTrans, trans, rot(makeVector(0.0, 0, 0, 0, 0, 0));
bool redraw_big_view = false;


// ros bag record jpeg compreseed images
#define ROSBAG_COMPRESSED_IMAGE 1

// ros bag
#if ROSBAG_COMPRESSED_IMAGE
const string rgb_img_topic = "/camera/rgb/image_color/compressed";
const string depth_img_topic = "/camera/depth/image_raw/compressedDepth";
#else
const string rgb_img_topic = "/camera/rgb/image_color";
const string depth_img_topic = "/camera/depth/image_raw";
#endif
rosbag::View *ptr_bag_filtered;
rosbag::View::const_iterator iter_bag_frame;
// flag of buffered depth frame has been processed
bool depth_buffer_processed;



// TODO: copy from compressed_depth_image_transport
// cannot fix linking issue
namespace compressed_depth_image_transport
{
sensor_msgs::Image::Ptr decodeCompressedDepthImage(const sensor_msgs::CompressedImage& message)
{

  cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);

  // Copy message header
  cv_ptr->header = message.header;

  // Assign image encoding
  std::string image_encoding = message.format.substr(0, message.format.find(';'));
  cv_ptr->encoding = image_encoding;

  // Decode message data
  if (message.data.size() > sizeof(ConfigHeader))
  {

    // Read compression type from stream
    ConfigHeader compressionConfig;
    memcpy(&compressionConfig, &message.data[0], sizeof(compressionConfig));

    // Get compressed image data
    const std::vector<uint8_t> imageData(message.data.begin() + sizeof(compressionConfig), message.data.end());

    // Depth map decoding
    float depthQuantA, depthQuantB;

    // Read quantization parameters
    depthQuantA = compressionConfig.depthParam[0];
    depthQuantB = compressionConfig.depthParam[1];

    if (sensor_msgs::image_encodings::bitDepth(image_encoding) == 32)
    {
      cv::Mat decompressed;
      try
      {
        // Decode image data
        decompressed = cv::imdecode(imageData, cv::IMREAD_UNCHANGED);
      }
      catch (cv::Exception& e)
      {
        //ROS_ERROR("%s", e.what());
        return sensor_msgs::Image::Ptr();
      }

      size_t rows = decompressed.rows;
      size_t cols = decompressed.cols;

      if ((rows > 0) && (cols > 0))
      {
        cv_ptr->image = cv::Mat(rows, cols, CV_32FC1);

        // Depth conversion
        cv::MatIterator_<float> itDepthImg = cv_ptr->image.begin<float>(),
                            itDepthImg_end = cv_ptr->image.end<float>();
        cv::MatConstIterator_<unsigned short> itInvDepthImg = decompressed.begin<unsigned short>(),
                                          itInvDepthImg_end = decompressed.end<unsigned short>();

        for (; (itDepthImg != itDepthImg_end) && (itInvDepthImg != itInvDepthImg_end); ++itDepthImg, ++itInvDepthImg)
        {
          // check for NaN & max depth
          if (*itInvDepthImg)
          {
            *itDepthImg = depthQuantA / ((float)*itInvDepthImg - depthQuantB);
          }
          else
          {
            *itDepthImg = std::numeric_limits<float>::quiet_NaN();
          }
        }

        // Publish message to user callback
        return cv_ptr->toImageMsg();
      }
    }
    else
    {
      // Decode raw image
      try
      {
        cv_ptr->image = cv::imdecode(imageData, CV_LOAD_IMAGE_UNCHANGED);
      }
      catch (cv::Exception& e)
      {
        //ROS_ERROR("%s", e.what());
        return sensor_msgs::Image::Ptr();
      }

      size_t rows = cv_ptr->image.rows;
      size_t cols = cv_ptr->image.cols;

      if ((rows > 0) && (cols > 0))
      {
        // Publish message to user callback
        return cv_ptr->toImageMsg();
      }
    }
  }
  return sensor_msgs::Image::Ptr();
}
}



void idle(void){

  if (depth_buffer_processed) {

    // check wether iter reach the end
    if (iter_bag_frame != ptr_bag_filtered->end()) {
      const rosbag::MessageInstance& message = *iter_bag_frame;

#if ROSBAG_COMPRESSED_IMAGE
      sensor_msgs::CompressedImage::ConstPtr img_ptr = message.instantiate<sensor_msgs::CompressedImage>();
      if (img_ptr != NULL) {

        if (message.getTopic() == depth_img_topic) {
          // depth image
          cout << "in depth frame" << endl;
          // convert to sensor msg Image
          sensor_msgs::Image::Ptr decodec_img_ptr =
              compressed_depth_image_transport::decodeCompressedDepthImage(*img_ptr);

          // check image property 640 x 480 x uchar16 x 1
          if (decodec_img_ptr->width == 640 && decodec_img_ptr->height == 480 &&
              decodec_img_ptr->step == 1280) {

            memcpy(depthImage.data(), &decodec_img_ptr->data[0],
                decodec_img_ptr->step*decodec_img_ptr->height);

            depth_buffer_processed = false;
            // rerun display func (do fusion)
            glutPostRedisplay();
          } else {
            cout << "decodec_img_ptr->width = " << decodec_img_ptr->width << endl;
            cout << "decodec_img_ptr->height = " << decodec_img_ptr->height << endl;
            cout << "decodec_img_ptr->step = " << decodec_img_ptr->step << endl;
            cout << "[ERROR] depth image format wrong" << endl;
          }

        } else if (message.getTopic() == rgb_img_topic) {
          // rgb image
          cout << "in color frame" << endl;
          // convert to opencv image
          cv::Mat cvimg = cv::imdecode(cv::Mat(img_ptr->data), CV_LOAD_IMAGE_COLOR);

          // check image property 640 x 480 x uchar16 x 1
          if (cvimg.cols == 640 && cvimg.rows == 480 &&
              cvimg.type() == CV_8UC3) {

            memcpy(depthImage.data(), cvimg.data, cvimg.cols*cvimg.rows*3);

          } else {
            cout << "[ERROR] depth image format wrong" << endl;
          }

        } else {
          //cout << "[ERROR] cannot identify Image topic type" << endl;
        }

      } else {
        //cout << "[ERROR] cannot find message type" << endl;
      }

#else
      sensor_msgs::Image::ConstPtr img_ptr = message.instantiate<sensor_msgs::Image>();
      if (img_ptr != NULL) {

        if (message.getTopic() == depth_img_topic) {
          // depth image
          cout << "in depth frame" << endl;
          // check image property 640 x 480 x uchar16 x 1
          if (img_ptr->width == 640 && img_ptr->height == 480 &&
              img_ptr->step == 1280) {

            memcpy(depthImage.data(), &img_ptr->data[0], img_ptr->step*img_ptr->height);

            depth_buffer_processed = false;
            // rerun display func (do fusion)
            glutPostRedisplay();
          } else {
            cout << "[ERROR] depth image format wrong" << endl;
          }

        } else if (message.getTopic() == rgb_img_topic) {
          // rgb image
          cout << "in color frame" << endl;
          // check image property 640 x 480 x uchar8 x 3
          if (img_ptr->width == 640 && img_ptr->height == 480 &&
              img_ptr->step == 1920) {

            memcpy(rgbImage.data(), &img_ptr->data[0], img_ptr->step*img_ptr->height);

          } else {
            cout << "[ERROR] depth image format wrong" << endl;
          }

        } else {
          //cout << "[ERROR] cannot identify Image topic type" << endl;
        }

      } else {
        //cout << "[ERROR] cannot find message type" << endl;
      }
#endif
      iter_bag_frame++;
    }

  }
}


void display(void){
  cout << "in" << endl;
  const uint2 imageSize = kfusion.configuration.inputSize;
  static bool integrate = true;

  glClear( GL_COLOR_BUFFER_BIT );
  const double startFrame = Stats.start();
  const double startProcessing = Stats.sample("kinect");

  kfusion.setKinectDeviceDepth(depthImage.getDeviceImage());
  Stats.sample("raw to cooked");

  integrate = kfusion.Track();
  Stats.sample("track");

  if((should_integrate && integrate && ((counter % integration_rate) == 0)) || reset){
    kfusion.Integrate();
    kfusion.Raycast();
    Stats.sample("integrate");
    if(counter > 2) // use the first two frames to initialize
      reset = false;
  }

  renderLight( lightScene.getDeviceImage(), kfusion.inputVertex[0], kfusion.inputNormal[0], light, ambient );
  renderLight( lightModel.getDeviceImage(), kfusion.vertex, kfusion.normal, light, ambient);
  renderTrackResult(trackModel.getDeviceImage(), kfusion.reduction);
  static int count = 4;
  if(count > 3 || redraw_big_view){
    renderInput( pos, normals, dep, kfusion.integration, toMatrix4( trans * rot * preTrans ) * getInverseCameraMatrix(kfusion.configuration.camera * 2), kfusion.configuration.nearPlane, kfusion.configuration.farPlane, kfusion.configuration.stepSize(), 0.75 * kfusion.configuration.mu);
    count = 0;
    redraw_big_view = false;
  } else
    count++;
  if(render_texture)
    renderTexture( texModel.getDeviceImage(), pos, normals, rgbImage.getDeviceImage(), getCameraMatrix(2*kfusion.configuration.camera) * inverse(kfusion.pose), light);
  else
    renderLight( texModel.getDeviceImage(), pos, normals, light, ambient);
  cudaDeviceSynchronize();

  Stats.sample("render");

  glClear(GL_COLOR_BUFFER_BIT);
  glRasterPos2i(0, 0);
  glDrawPixels(lightScene);
  glRasterPos2i(0, 240);
  glPixelZoom(0.5, -0.5);
  glDrawPixels(rgbImage);
  glPixelZoom(1,-1);
  glRasterPos2i(320,0);
  glDrawPixels(lightModel);
  glRasterPos2i(320,240);
  glDrawPixels(trackModel);
  glRasterPos2i(640, 0);
  glDrawPixels(texModel);
  const double endProcessing = Stats.sample("draw");

  Stats.sample("total", endProcessing - startFrame, PerfStats::TIME);
  Stats.sample("total_proc", endProcessing - startProcessing, PerfStats::TIME);

  if(printCUDAError())
    exit(1);

  ++counter;

  if(counter % 50 == 0){
    Stats.print();
    Stats.reset();
    cout << endl;
  }

  // processed frame
  depth_buffer_processed = true;

  glutSwapBuffers();
  cout << "out" << endl;
}


void keys(unsigned char key, int x, int y){
  switch(key){
    case 'c':
      kfusion.Reset();
      kfusion.setPose(toMatrix4(initPose));
      reset = true;
      break;
    case 'q':
      exit(0);
      break;
    case 'i':
      should_integrate = !should_integrate;
      break;
    case 't':
      render_texture = !render_texture;
      break;
  }
}

void specials(int key, int x, int y){
  switch(key){
    case GLUT_KEY_LEFT:
      rot = SE3<float>(makeVector(0.0, 0, 0, 0, 0.1, 0)) * rot;
      break;
    case GLUT_KEY_RIGHT:
      rot = SE3<float>(makeVector(0.0, 0, 0, 0, -0.1, 0)) * rot;
      break;
    case GLUT_KEY_UP:
      rot *= SE3<float>(makeVector(0.0, 0, 0, -0.1, 0, 0));
      break;
    case GLUT_KEY_DOWN:
      rot *= SE3<float>(makeVector(0.0, 0, 0, 0.1, 0, 0));
      break;
  }
  redraw_big_view = true;
}

void reshape(int width, int height){
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glViewport(0, 0, width, height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glColor3f(1.0f,1.0f,1.0f);
  glRasterPos2f(-1, 1);
  glOrtho(-0.375, width-0.375, height-0.375, -0.375, -1 , 1); //offsets to make (0,0) the top left pixel (rather than off the display)
  glPixelZoom(1,-1);
}

void exitFunc(void){
  delete ptr_bag_filtered;
  //CloseKinect();
  kfusion.Clear();
  cudaDeviceReset();
}

int main(int argc, char ** argv) {

  // load and filter rosbag file by topics
  const string bagfile_name = "/home/jing/data/kinect/2016-12-21-12-35-47_0.bag";

  rosbag::Bag bag;
  bag.open(bagfile_name, rosbag::bagmode::Read);

  vector<string> topics;
  topics.push_back(rgb_img_topic);
  topics.push_back(depth_img_topic);

  ptr_bag_filtered = new rosbag::View(bag, rosbag::TopicQuery(topics));

  // iterator to first frame
  iter_bag_frame = ptr_bag_filtered->begin();

  // set true let view feeds in first frame
  depth_buffer_processed = true;

  // volume size
  const float size = (argc > 1) ? atof(argv[1]) : 2.f;

  KFusionConfig config;

  // it is enough now to set the volume resolution once.
  // everything else is derived from that.
  // config.volumeSize = make_uint3(64);
  // config.volumeSize = make_uint3(128);
  config.volumeSize = make_uint3(256);

  // these are physical dimensions in meters
  config.volumeDimensions = make_float3(size);
  config.nearPlane = 0.4f;
  config.farPlane = 5.0f;
  config.mu = 0.1;
  config.combinedTrackAndReduce = false;

  // change the following parameters for using 640 x 480 input images
  config.inputSize = make_uint2(320,240);
  config.camera =  make_float4(531.15/2, 531.15/2, 640/4, 480/4);

  // config.iterations is a vector<int>, the length determines
  // the number of levels to be used in tracking
  // push back more then 3 iteraton numbers to get more levels.
  config.iterations[0] = 10;
  config.iterations[1] = 5;
  config.iterations[2] = 4;

  config.dist_threshold = (argc > 2 ) ? atof(argv[2]) : config.dist_threshold;
  config.normal_threshold = (argc > 3 ) ? atof(argv[3]) : config.normal_threshold;

  initPose = SE3<float>(makeVector(size/2, size/2, 0, 0, 0, 0));

  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE );
  glutInitWindowSize(config.inputSize.x * 2 + 640, max(config.inputSize.y * 2, 480));
  glutCreateWindow("kfusion");

  kfusion.Init(config);

  // input buffers
  depthImage.alloc(make_uint2(640, 480));
  rgbImage.alloc(make_uint2(640, 480));

  // render buffers
  lightScene.alloc(config.inputSize), trackModel.alloc(config.inputSize), lightModel.alloc(config.inputSize);
  pos.alloc(make_uint2(640, 480)), normals.alloc(make_uint2(640, 480)), dep.alloc(make_uint2(640, 480)), texModel.alloc(make_uint2(640, 480));

  if(printCUDAError()) {
    delete ptr_bag_filtered;
    cudaDeviceReset();
    return 1;
  }

  memset(depthImage.data(), 0, depthImage.size.x*depthImage.size.y * sizeof(uint16_t));
  memset(rgbImage.data(), 0, rgbImage.size.x*rgbImage.size.y * sizeof(uchar3));

  /*
  uint16_t * buffers[2] = {depthImage[0].data(), depthImage[1].data()};

  if(InitKinect(buffers, (unsigned char *)rgbImage.data())){
    cudaDeviceReset();
    return 1;
  }
  */

  kfusion.setPose(toMatrix4(initPose));

  // model rendering parameters
  preTrans = SE3<float>::exp(makeVector(0.0, 0, -size, 0, 0, 0));
  trans = SE3<float>::exp(makeVector(0.5, 0.5, 0.5, 0, 0, 0) * size);

  atexit(exitFunc);
  glutDisplayFunc(display);
  glutKeyboardFunc(keys);
  glutSpecialFunc(specials);
  glutReshapeFunc(reshape);
  glutIdleFunc(idle);

  cout << "ready" << endl;
  glutMainLoop();

  delete ptr_bag_filtered;
  //CloseKinect();

  return 0;
}


