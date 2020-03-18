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

  std::vector<float> log_table_;

  std::vector<int> map_idx_;

  float logCount(int count);

 public:
  FeatureGenerator() {}
  ~FeatureGenerator() {}

  bool init(float *in_feature_ptr, int range, int width, int height,
            const bool use_constant_feature, const bool use_intensity_feature);

  void generate(const pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_ptr,
                float *max_height_data,
                const bool use_constant_feature,
                const bool use_intensity_feature);
};

#endif  // FEATURE_GENERATOR_H
