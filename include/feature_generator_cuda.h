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
#ifndef FEATURE_GENERATOR_CUDA_H
#define FEATURE_GENERATOR_CUDA_H

#include <pcl/point_types.h>
// #include <pcl_ros/point_cloud.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl_ros/transforms.h>
#include <opencv2/core/core.hpp>

#include "util.h"
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

class FeatureGeneratorCuda {
 private:
  int width_ = 640;
  int height_ = 640;
  int range_ = 0;

  // raw feature data
  float* max_height_data_ = nullptr;
  float* mean_height_data_ = nullptr;
  float* count_data_ = nullptr;
  float* direction_data_ = nullptr;
  float* top_intensity_data_ = nullptr;
  float* mean_intensity_data_ = nullptr;
  float* distance_data_ = nullptr;
  float* nonempty_data_ = nullptr;

  // float* max_height_data_;
  // float* mean_height_data_;
  // float* count_data_;
  // float* direction_data_;
  // float* top_intensity_data_;
  // float* mean_intensity_data_;
  // float* distance_data_;
  // float* nonempty_data_;

  pcl::PointXYZI* pc_gpu_ = nullptr;
  std::vector<int> map_idx_;
  // int* map_idx_;
  int* map_idx_gpu_ = nullptr;
  int pc_gpu_size_ = 0;


  // std::vector<float> in_feature_;


  const int kMaxPointCloudGPUSize = 120000;
  const int kGPUThreadSize = 512;

  float min_height_ = 0.0;
  float max_height_ = 0.0;

  std::vector<float> log_table_;


  float logCount(int count);

 public:
  FeatureGeneratorCuda() {}
  ~FeatureGeneratorCuda() {}

  bool init(int range, int width, int height,
            const bool use_constant_feature, const bool use_intensity_feature);

  void generate(const pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_ptr,
                float *max_height_data,
                const bool use_constant_feature,
                const bool use_intensity_feature);

  float in_feature_[640*640*8];
};

#endif  // FEATURE_GENERATOR_CUDA_H
