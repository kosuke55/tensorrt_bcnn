#include <ros/package.h>
#include <boost/filesystem.hpp>

#include <algorithm>
#include <string>
#include "tensorrt_bcnn_ros.h"

#include <cuda.h>
#include <chrono>

TensorrtBcnnROS::TensorrtBcnnROS(/* args */) : pnh_("~") {}
bool TensorrtBcnnROS::init() {
  rows_ = 640;
  cols_ = 640;
  siz_ = rows_ * cols_;
  in_feature.resize(siz_ * 8);

  std::string package_path = ros::package::getPath("tensorrt_bcnn");
  std::string engine_path = package_path + "/data/bcnn_0111.engine";

  std::ifstream fs(engine_path);
  if (fs.is_open()) {
    ROS_INFO("load %s", engine_path.c_str());
    net_ptr_.reset(new Tn::trtNet(engine_path));
  } else {
    ROS_INFO("Could not find %s.", engine_path.c_str());
  }
  feature_generator_.reset(new FeatureGenerator());
  if (!feature_generator_->init(&in_feature[0])) {
    ROS_ERROR("[%s] Fail to Initialize feature generator for CNNSegmentation",
              __APP_NAME__);
    return false;
  }

  ros::NodeHandle private_node_handle("~");  // to receive args
  private_node_handle.param<std::string>("points_src", topic_src_,
                                         "/points_raw");
  ROS_INFO("[%s] points_src: %s", __APP_NAME__, topic_src_.c_str());
}

TensorrtBcnnROS::~TensorrtBcnnROS() {}

void TensorrtBcnnROS::createROSPubSub() {
  sub_points_ =
      nh_.subscribe(topic_src_, 1, &TensorrtBcnnROS::pointsCallback, this);

  pub_objects_ =
      nh_.advertise<autoware_msgs::DynamicObjectWithFeatureArray>("rois", 1);
  pub_image_ = nh_.advertise<sensor_msgs::Image>(
      "/perception/tensorrt_bcnn/classified_image", 1);
  confidence_pub_ = nh_.advertise<sensor_msgs::Image>("confidence_image", 1);
  confidence_map_pub_ =
      nh_.advertise<nav_msgs::OccupancyGrid>("confidence_map", 1);
}

void TensorrtBcnnROS::pointsCallback(const sensor_msgs::PointCloud2 &msg) {

  pcl::PointCloud<pcl::PointXYZI>::Ptr in_pc_ptr(
      new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromROSMsg(msg, *in_pc_ptr);
  pcl::PointIndices valid_idx;
  auto &indices = valid_idx.indices;
  indices.resize(in_pc_ptr->size());
  std::iota(indices.begin(), indices.end(), 0);
  message_header_ = msg.header;
  for (int i = 0; i < siz_ * 8; ++i) {
    if (i < siz_) {
      in_feature[i] = -5;
    } else if (siz_ <= i && i < siz_ * 3) {
      in_feature[i] = 0;
    } else if (siz_ * 4 <= i && i < siz_ * 6) {
      in_feature[i] = 0;
    } else if (siz_ * 7 <= i && i < siz_ * 8) {
      in_feature[i] = 0;
    }
  }

  feature_generator_->generate(in_pc_ptr, &in_feature[0]);

  int outputCount = net_ptr_->getOutputSize() / sizeof(float);
  std::unique_ptr<float[]> output_data(new float[outputCount]);

  cv::Mat confidence_image(640, 640, CV_8UC1);
  for (int row = 0; row < 640; ++row) {
    unsigned char *src = confidence_image.ptr<unsigned char>(row);
    for (int col = 0; col < 640; ++col) {
      int grid = row + col * 640;
      if (output[grid] > 0.5) {
        // if (in_feature[siz_ * 7 + grid] > 0.5) {
        src[cols_ - col - 1] = 255;
      } else {
        src[cols_ - col - 1] = 0;
      }
    }
  }
  confidence_pub_.publish(
      cv_bridge::CvImage(message_header_, sensor_msgs::image_encodings::MONO8,
                         confidence_image)
          .toImageMsg());

  nav_msgs::OccupancyGrid confidence_map;
  confidence_map.header = message_header_;
  confidence_map.header.frame_id = "velodyne";
  confidence_map.info.width = 640;
  confidence_map.info.height = 640;
  confidence_map.info.origin.orientation.w = 1;
  double resolution = 120. / 640.;
  confidence_map.info.resolution = resolution;
  confidence_map.info.origin.position.x = -((640 + 1) * resolution * 0.5f);
  confidence_map.info.origin.position.y = -((640 + 1) * resolution * 0.5f);

  // read the pixels of the image and fill the confidence_map table
  int data;
  for (int i = 640 - 1; i >= 0; i--) {
    for (unsigned int j = 0; j < 640; j++) {
      data = confidence_image.data[i * 640 + j];
      if (data >= 123 && data <= 131) {
        confidence_map.data.push_back(-1);
      } else if (data >= 251 && data <= 259) {
        confidence_map.data.push_back(0);
      } else
        confidence_map.data.push_back(100);
    }
  }
  confidence_map_pub_.publish(confidence_map);
}
