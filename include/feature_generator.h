/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FEATURE_GENERATOR_H
#define FEATURE_GENERATOR_H

#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <opencv2/core/core.hpp>

#include "util.h"

class FeatureGenerator {
 private:
  int width_ = 640;
  int height_ = 640;
  int range_ = 0;

  float min_height_ = 0.0;
  float max_height_ = 0.0;

  // raw feature data
  // std::vector<float> max_height_data_;
  // std::vector<float> mean_height_data_;
  // std::vector<float> count_data_;
  // std::vector<float> direction_data_;
  // std::vector<float> top_intensity_data_;
  // std::vector<float> mean_intensity_data_;
  // std::vector<float> distance_data_;
  // std::vector<float> nonempty_data_;

  // float* max_height_data_ = nullptr;
  // float* mean_height_data_ = nullptr;
  // float* count_data_ = nullptr;
  // float* direction_data_ = nullptr;
  // float* top_intensity_data_ = nullptr;
  // float* mean_intensity_data_ = nullptr;
  // float* distance_data_ = nullptr;
  // float* nonempty_data_ = nullptr;

  // output Caffe blob
  // caffe::Blob<float>* out_blob_ = nullptr;

  std::vector<float> log_table_;

  // point index in feature map
  std::vector<int> map_idx_;

  float logCount(int count);

 public:
  // FeatureGenerator() : max_height_data_(width_ * height_, -5),
  //                      mean_height_data_(width_ * height_, 0),
  //                      count_data_(width_ * height_, 0),
  //                      direction_data_(width_ * height_, 0),
  //                      top_intensity_data_(width_ * height_, 0),
  //                      mean_intensity_data_(width_ * height_, 0),
  //                      distance_data_(width_ * height_, 0),
  //                      nonempty_data_(width_ * height_, 0) {}
  FeatureGenerator() {}
  ~FeatureGenerator() {}

  // bool init(caffe::Blob<float>* out_blob);
  bool init(float *in_feature_ptr);
  void generate(const pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_ptr,
                float *max_height_data_);
  // cv::Mat& in_feature);
};

#endif  // FEATURE_GENERATOR_H
