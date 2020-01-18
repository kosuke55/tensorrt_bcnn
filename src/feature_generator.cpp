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

#include "feature_generator.h"

bool FeatureGenerator::init(float *in_feature_ptr) {
  range_ = 60;
  width_ = 640;
  height_ = 640;
  min_height_ = -5.0;
  max_height_ = 5.0;

  log_table_.resize(256);
  for (size_t i = 0; i < log_table_.size(); ++i) {
    log_table_[i] = std::log1p(static_cast<float>(i));
  }

  int channel_index = 0;
  int siz = height_ * width_;

  float *direction_data_, *distance_data_;
  direction_data_ = in_feature_ptr + siz * 3;
  distance_data_ = in_feature_ptr + siz * 6;

  for (int row = 0; row < height_; ++row) {
    for (int col = 0; col < width_; ++col) {
      int idx = row * width_ + col;
      // * row <-> x, column <-> y
      // retutn the distance from my car to center of the grid.
      // Pc means point cloud = real world scale. so transform pixel scale to
      // real world scale
      float center_x = Pixel2Pc(row, height_, range_);
      float center_y = Pixel2Pc(col, width_, range_);
      constexpr double K_CV_PI = 3.1415926535897932384626433832795;
      // normaliztion. -0.5~0.5
      direction_data_[idx] =
          static_cast<float>(std::atan2(center_y, center_x) / (2.0 * K_CV_PI));
      distance_data_[idx] =
          static_cast<float>(std::hypot(center_x, center_y) / 60.0 - 0.5);
    }
  }

  return true;
}

float FeatureGenerator::logCount(int count) {
  if (count < static_cast<int>(log_table_.size())) {
    return log_table_[count];
  }
  return std::log(static_cast<float>(1 + count));
}

void FeatureGenerator::generate(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_ptr,
    float *max_height_data_) {
  const auto &points = pc_ptr->points;
  int siz = height_ * width_;
  float *mean_height_data_, *count_data_, *top_intensity_data_,
      *mean_intensity_data_, *nonempty_data_;

  mean_height_data_ = max_height_data_ + siz;
  count_data_ = max_height_data_ + siz * 2;
  top_intensity_data_ = max_height_data_ + siz * 4;
  mean_intensity_data_ = max_height_data_ + siz * 5;
  nonempty_data_ = max_height_data_ + siz * 7;

  map_idx_.resize(points.size());
  float inv_res_x =
      0.5 * static_cast<float>(width_) / static_cast<float>(range_);
  float inv_res_y =
      0.5 * static_cast<float>(height_) / static_cast<float>(range_);

  for (size_t i = 0; i < points.size(); ++i) {
    if (points[i].z <= min_height_ || points[i].z >= max_height_) {
      map_idx_[i] = -1;
      continue;
    }
    // project point cloud to 2d map. clac in which grid point is.
    // * the coordinates of x and y are exchanged here
    // (row <-> x, column <-> y)
    int pos_x = F2I(points[i].y, range_, inv_res_x);
    int pos_y = F2I(points[i].x, range_, inv_res_y);
    if (pos_x >= width_ || pos_x < 0 || pos_y >= height_ || pos_y < 0) {
      map_idx_[i] = -1;
      continue;
    }
    map_idx_[i] = pos_y * width_ + pos_x;
    int idx = map_idx_[i];
    float pz = points[i].z;
    float pi = points[i].intensity / 255.0;
    if (max_height_data_[idx] < pz) {
      max_height_data_[idx] = pz;
      top_intensity_data_[idx] = pi;  // not I_max but I of z_max ?
    }
    mean_height_data_[idx] += static_cast<float>(pz);
    mean_intensity_data_[idx] += static_cast<float>(pi);
    count_data_[idx] += static_cast<float>(1);
  }
  for (int i = 0; i < siz; ++i) {
    constexpr double EPS = 1e-6;
    if (count_data_[i] < EPS) {
      max_height_data_[i] = static_cast<float>(0);
    } else {
      mean_height_data_[i] /= count_data_[i];
      mean_intensity_data_[i] /= count_data_[i];
      nonempty_data_[i] = static_cast<float>(1);
    }
    count_data_[i] = logCount(static_cast<int>(count_data_[i]));
  }
}
