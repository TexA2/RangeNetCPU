#pragma once

// standard stuff
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <cmath>
#include <memory>

// yaml
#include "yaml-cpp/yaml.h"

// plc
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

// libtorch (pyTorch) for CPU
#include <torch/torch.h>

// onnxruntime for CPU
#include <onnxruntime_cxx_api.h>

template <typename T>
  std::vector<size_t> sort_indexes(const std::vector<T> &v) {

    // initialize original index locations
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v. >: decrease <: increase
    std::sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});

    return idx;
  }    


struct RangeImageParams {
    float fov_up = 15.0f;    // Верхний угол обзора (град)
    float fov_down = -15.0f; // Нижний угол обзора (град)
    int width = 2048;        // Ширина изображения (пикс)
    int height = 64;         // Высота изображения (пикс)
    float max_range = 100.0f; // Максимальная дальность (м)
};

template <typename T>
struct VirtualCenter {
  T x;
  T y;
  T z;
};



namespace rangenet {
  namespace segmentation {


  class Net {
  public:
    typedef std::tuple< u_char, u_char, u_char> color;

    Net(const std::string& model_path);

    virtual ~Net(){};

    void getPoints(const std::string& cloud_path);

    std::vector<std::vector<float>> doProjection_origin(bool yaml = true);

    std::vector<int> getLabelMap() { return _lable_map;}
    std::map<uint32_t, color> getColorMap() { return _color_map;}

    void init_model();

    std::vector<std::vector<float>> infer();


    std::vector<std::array<uint8_t, 3>> getLabels(const std::vector<std::vector<float>>& semantic_scan,
                                                  const uint32_t& num_points);


    void convertToPointCloud(const std::vector<std::vector<float>>& range_imag);                                   

    void convertToPointCloudColor(const std::vector<std::array<uint8_t, 3>>& colors,
                                  const std::vector<std::vector<float>>& range_image);

    void getData();

  protected:
  
    // general
    std::string _model_path;  // Where to get model weights and cfg

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
    Ort::Env _env;
    std::unique_ptr<Ort::Session> _session;

    // image properties
    int _img_h, _img_w, _img_d;  // height, width, and depth for inference

    std::vector<float> _img_means, _img_stds; // mean and std per channel

    // problem properties
    int32_t _n_classes;  // number of classes to differ from
    
    // sensor properties
    float _fov_up, _fov_down; // field of view up and down in radians

    std::vector<float> proj_xs; // stope a copy in original order
    std::vector<float> proj_ys;

    // config
    YAML::Node data_cfg;  // yaml nodes with configuration from training
    YAML::Node arch_cfg;  // yaml nodes with configuration from training

    std::vector<int> _lable_map;
    std::map<uint32_t, color> _color_map;
    std::map<uint32_t, color> _argmax_to_rgb;  // for color conversion


    std::vector<float> invalid_input =  {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    std::vector<float> invalid_output = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                         0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                         0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                         0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  };

  }  // namespace segmentation
}  // namespace rangenet
