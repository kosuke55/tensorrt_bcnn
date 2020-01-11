#include "tensorrt_bcnn_ros.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "tensorrt_bcnn");
  TensorrtBcnnROS node;
  node.init();
  node.createROSPubSub();
  ros::spin();

  return 0;
}
