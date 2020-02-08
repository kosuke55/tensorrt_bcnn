#ifndef _DATA_READER_H_
#define _DATA_READER_H_

#include <list>
#include <string>
#include <vector>

namespace Tn {
std::list<std::string> readFileList(const std::string& fileName);

struct Source {
  std::string fileName;
  int label;
};
std::list<Source> readLabelFileList(const std::string& fileName);

struct Bbox {
  int classId;
  int left;
  int right;
  int top;
  int bot;
  float score;
};
//[lst<filename>,lst<bbox_vec>]
std::tuple<std::list<std::string>, std::list<std::vector<Bbox>>>
readObjectLabelFileList(const std::string& fileName);
}

#endif