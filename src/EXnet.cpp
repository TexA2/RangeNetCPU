#include "net.hpp"


void calculateMean(const std::vector<std::vector<float>>& inputs)
{
  const int num_feature = 5; // range, x, y, z, intensity

  std::cout << inputs.size() << std::endl;

  std::vector<float>means(num_feature, 0);
  std::vector<float>stds (num_feature, 0);

  for (const auto& point : inputs)
  {
    for(int i = 0; i < num_feature; ++i)
      means[i] += point[i];
  }


  for(int i = 0; i < num_feature; ++i)
    means[i] /= inputs.size();

  for (const auto& point : inputs)
  {
    for(int i = 0; i < num_feature; ++i)
      stds[i] += std::pow(point[i] - means[i], 2);
  }  

  for (int i = 0; i < num_feature; ++i) 
    stds[i] = std::sqrt(stds[i] / inputs.size());

  std::cout << "_img_means" << std::endl;
  for (const auto& i : means)
    std::cout << i << " ";
  std::cout << std::endl;


  std::cout << "_img_stds" << std::endl;
    for (const auto& i : stds)
    std::cout << i << " ";
  std::cout << std::endl;
}

namespace rangenet {
  namespace segmentation {

  Net::Net(const std::string& model_path): _model_path(model_path)
  {
      // Try to get the config file as well
      std::string arch_cfg_path = _model_path + "/arch_cfg.yaml";
      try {
        arch_cfg = YAML::LoadFile(arch_cfg_path);
      } catch (YAML::Exception& ex) {
        throw std::runtime_error("Can't open cfg.yaml from " + arch_cfg_path);
      }

      // Assign fov_up and fov_down from arch_cfg
      _fov_up = arch_cfg["dataset"]["sensor"]["fov_up"].as<double>();
      _fov_down = arch_cfg["dataset"]["sensor"]["fov_down"].as<double>();

      std::string data_cfg_path = _model_path + "/data_cfg.yaml";
      try {
        data_cfg = YAML::LoadFile(data_cfg_path);
      } catch (YAML::Exception& ex) {
        throw std::runtime_error("Can't open cfg.yaml from " + data_cfg_path);
      }

      // Get label dictionary from yaml cfg
      YAML::Node color_map;
      try {
        color_map = data_cfg["color_map"];
      } catch (YAML::Exception& ex) {
        std::cerr << "Can't open one the label dictionary from cfg in " + data_cfg_path
                  << std::endl;
        throw ex;
      }

      // Generate string map from xentropy indexes (that we'll get from argmax)
      YAML::const_iterator it;

      for (it = color_map.begin(); it != color_map.end(); ++it) {
        // Get label and key
        int key = it->first.as<int>();  // <- key
        Net::color color = std::make_tuple(
            static_cast<u_char>(color_map[key][0].as<unsigned int>()),
            static_cast<u_char>(color_map[key][1].as<unsigned int>()),
            static_cast<u_char>(color_map[key][2].as<unsigned int>()));
        _color_map[key] = color;
      }

      // Get learning class labels from yaml cfg
      YAML::Node learning_class;
      try {
        learning_class = data_cfg["learning_map_inv"];
      } catch (YAML::Exception& ex) {
        std::cerr << "Can't open one the label dictionary from cfg in " + data_cfg_path
                  << std::endl;
        throw ex;
      }

      // get the number of classes
      _n_classes = learning_class.size();

      // remapping the colormap lookup table
      _lable_map.resize(_n_classes);
      for (it = learning_class.begin(); it != learning_class.end(); ++it) {
        int key = it->first.as<int>();  // <- key
        _argmax_to_rgb[key] = _color_map[learning_class[key].as<unsigned int>()];
        _lable_map[key] = learning_class[key].as<unsigned int>();
      }

      // get image size
      _img_h = arch_cfg["dataset"]["sensor"]["img_prop"]["height"].as<int>();
      _img_w = arch_cfg["dataset"]["sensor"]["img_prop"]["width"].as<int>();
      _img_d = 5; // range, x, y, z, remission

      // get normalization parameters
        YAML::Node img_means, img_stds;
        try {
          img_means = arch_cfg["dataset"]["sensor"]["img_means"];
          img_stds = arch_cfg["dataset"]["sensor"]["img_stds"];
        } catch (YAML::Exception& ex) {
          std::cerr << "Can't open one the mean or std dictionary from cfg"
                    << std::endl;
          throw ex;
        }
        // fill in means from yaml node
        for (it = img_means.begin(); it != img_means.end(); ++it) {
          // Get value
          float mean = it->as<float>();
          // Put in indexing vector
          _img_means.push_back(mean);
        }
        // fill in stds from yaml node
        for (it = img_stds.begin(); it != img_stds.end(); ++it) {
          // Get value
          float std = it->as<float>();
          // Put in indexing vector
          _img_stds.push_back(std);
        }

    cloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
}
 
  void Net::getPoints(const std::string& cloud_path)
  {
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(cloud_path + "/cloud.pcd", *cloud) == -1) {
        std::cerr << "Error: Could not read PCD file!" << std::endl;
    }
  }


  void Net::getData()
  {
    _img_means = {19.6365, -1.56444, 2.4058, -0.615846 ,0};
    _img_stds = {16.7977, 18.2008, 18.0324, 1.59119, 0 };

    float min_z = std::numeric_limits<float>::max();
    float max_z = -std::numeric_limits<float>::max();

    for (const auto& p : cloud->points)
    {
        if (p.z < min_z) min_z = p.z;
        if (p.z > max_z) max_z = p.z;
    }

     _fov_down = std::numeric_limits<float>::max();
     _fov_up = -std::numeric_limits<float>::max();


    for (const auto& p : cloud->points) 
    {
    float xy_dist = sqrt(p.x * p.x + p.y * p.y);
    float pitch = atan2(p.z, xy_dist);  // угол места
    
    if (p.z <= min_z + 0.01f) {  // учитываем точки близкие к min_z
        _fov_down = std::min(_fov_down, pitch);
    }
    if (p.z >= max_z - 0.01f) {  // учитываем точки близкие к max_z
        _fov_up = std::max(_fov_up, pitch);
    }
    }
}



  std::vector<std::vector<float>> Net::doProjection_origin(bool yaml)
  {

    if (!yaml) getData();

    float fov_up = _fov_up;    // field of view up in radians
    float fov_down = _fov_down;  // field of view down in radians
    float fov = std::abs(fov_down) + std::abs(fov_up); // get field of view total in radians
    //float fov = fov_up - fov_down; // get field of view total in radians


    std::vector<float> ranges;
    std::vector<float> xs;
    std::vector<float> ys;
    std::vector<float> zs;
    std::vector<float> intensitys;

    std::vector<float> proj_xs_tmp;
    std::vector<float> proj_ys_tmp;

    for (const auto& p : cloud->points) 
    {
      float x = p.x;
      float y = p.y;
      float z = p.z;
      float intensity = p.intensity;
      float range = std::sqrt(x*x+y*y+z*z);
      ranges.push_back(range);
      xs.push_back(x);
      ys.push_back(y);
      zs.push_back(z);
      intensitys.push_back(intensity);

      // get angles
      float yaw = -std::atan2(y, x);
      float pitch = std::asin(z / range);

      // get projections in image coords
      float proj_x = 0.5 * (yaw / M_PI + 1.0); // in [0.0, 1.0]
      float proj_y = 1.0 - (pitch + std::abs(fov_down)) / fov; // in [0.0, 1.0]

      // scale to image size using angular resolution
      proj_x *= _img_w; // in [0.0, W]
      proj_y *= _img_h; // in [0.0, H]

      // round and clamp for use as index
      proj_x = std::floor(proj_x);
      proj_x = std::min(_img_w - 1.0f, proj_x);
      proj_x = std::max(0.0f, proj_x); // in [0,W-1]
      proj_xs_tmp.push_back(proj_x);

      proj_y = std::floor(proj_y);
      proj_y = std::min(_img_h - 1.0f, proj_y);
      proj_y = std::max(0.0f, proj_y); // in [0,H-1]
      proj_ys_tmp.push_back(proj_y);
    }

  // stope a copy in original order
  proj_xs = proj_xs_tmp;
  proj_ys = proj_ys_tmp;

  // order in decreasing depth
  std::vector<size_t> orders = sort_indexes(ranges);
  std::vector<float> sorted_proj_xs;
  std::vector<float> sorted_proj_ys;
  std::vector<std::vector<float>> inputs;

  for (size_t idx : orders){
    sorted_proj_xs.push_back(proj_xs[idx]);
    sorted_proj_ys.push_back(proj_ys[idx]);
    std::vector<float> input = {ranges[idx], xs[idx], ys[idx], zs[idx], intensitys[idx]};
    inputs.push_back(input);
  }

    calculateMean(inputs);

  std::vector<std::vector<float>> range_image(_img_w * _img_h, invalid_input);

  for (uint32_t i = 0; i < inputs.size(); ++i) {
    int x = static_cast<int>(sorted_proj_xs[i]);
    int y = static_cast<int>(sorted_proj_ys[i]);
    int idx = y * _img_w + x;
    range_image[idx] = inputs[i];
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr range_cloud(new pcl::PointCloud<pcl::PointXYZI>);
  range_cloud->reserve(_img_h * _img_w); 

    for (int y = 0; y < _img_h; ++y) {
      for (int x = 0; x < _img_w; ++x) {
                size_t linear_index = y * _img_w + x;
                if (range_image[linear_index] == invalid_input )continue;

                pcl::PointXYZI point;
                point.x = x;
                point.y = y;
                point.z = range_image[linear_index][0];  // Дальность
                point.intensity = range_image[linear_index][4];
                range_cloud->push_back(point);
        }
    }

    pcl::io::savePCDFileASCII("output.pcd", *range_cloud);
    std::cout << "Range View saved to " << "output.pcd" << std::endl;

    return range_image;
  }

  void Net::init_model()
  {
    _env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "RangeNet");

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    _session = std::make_unique<Ort::Session>(_env, (_model_path + "/model.onnx").c_str(), session_options);
  }


std::vector<std::vector<float>> Net::infer()
{
    // Project point clouds into range image
    std::vector<std::vector<float>> projected_data = doProjection_origin();

    // Prepare input tensor
    int channel_offset = _img_h * _img_w;
    std::vector<float> input_data(channel_offset * _img_d, 0.0f);
    std::vector<int> invalid_idxs;

    // Normalize and fill input data
    for (uint32_t pixel_id = 0; pixel_id < projected_data.size(); pixel_id++) {
        bool all_zeros = std::all_of(projected_data[pixel_id].begin(), 
                                   projected_data[pixel_id].end(), 
                                   [](float i) { return i == 0.0f; });
        
        if (all_zeros) {
            invalid_idxs.push_back(pixel_id);
        }

        for (int i = 0; i < _img_d; i++) {
            int buffer_idx = channel_offset * i + pixel_id;
            input_data[buffer_idx] = all_zeros ? 0.0f : 
                                    (projected_data[pixel_id][i] - _img_means[i]) / _img_stds[i];
        }
    }


    // Create input tensor
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, 
                                                             OrtMemType::OrtMemTypeDefault);
    
    std::vector<int64_t> input_shape = {1, _img_d, _img_h, _img_w};  // NCHW format

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, 
        input_data.data(), 
        input_data.size(), 
        input_shape.data(), 
        input_shape.size()
    );


    std::vector<std::string> input_name = _session->GetInputNames();
    std::vector<std::string> output_name = _session->GetOutputNames();

    // Run inference
    const char* input_names[] = {input_name[0].c_str()};
    const char* output_names[] = {output_name[0].c_str()};
    
    auto output_tensors = _session->Run(
        Ort::RunOptions{nullptr}, 
        input_names, 
        &input_tensor, 
        1, 
        output_names, 
        1
    );

    // Get output data
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    std::vector<std::vector<float>> range_image(channel_offset);

    for (int pixel_id = 0; pixel_id < channel_offset; pixel_id++) {
        for (int i = 0; i < _n_classes; i++) {
            int buffer_idx = channel_offset * i + pixel_id;
            range_image[pixel_id].push_back(output_data[buffer_idx]);
        }
    }

    // Set invalid pixels
    for (int idx : invalid_idxs) {
        range_image[idx] = invalid_output;
    }

    // Unprojection
    std::vector<std::vector<float>> semantic_scan;

    for (uint32_t i = 0; i < proj_ys.size(); i++) {
        semantic_scan.push_back(range_image[int(proj_ys[i] * _img_w + proj_xs[i])]);
    }

    return semantic_scan;
}


std::vector<std::array<uint8_t, 3>> Net::getLabels(const std::vector<std::vector<float>>& semantic_scan, 
                                                   const uint32_t& num_points) 
{
    std::vector<std::array<uint8_t, 3>> labels(num_points);
    std::vector<float> labels_prob(num_points, 0.0f);

    for (uint32_t i = 0; i < num_points; ++i) {
        for (int32_t j = 0; j < _n_classes; ++j) {
            if (labels_prob[i] <= semantic_scan[i][j]) {
                labels[i] = {
                    static_cast<uint8_t>(std::get<0>(_argmax_to_rgb[j])),
                    static_cast<uint8_t>(std::get<1>(_argmax_to_rgb[j])),
                    static_cast<uint8_t>(std::get<2>(_argmax_to_rgb[j]))
                };
                labels_prob[i] = semantic_scan[i][j];
            }
        }
    }
    return labels;
}


void Net::convertToPointCloud(const std::vector<std::array<uint8_t, 3>>& colors)
{
      std::vector<std::vector<float>> range_image = doProjection_origin();
    // Создаем облако точек
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud->resize(range_image.size());

int i = 0;
    for (int y = 0; y < _img_h; ++y) {
      for (int x = 0; x < _img_w; ++x) {
                size_t linear_index = y * _img_w + x;
                if (range_image[linear_index] == invalid_input )continue;

                pcl::PointXYZRGB point;
                point.x = x;
                point.y = y;    
                point.z = range_image[linear_index][0];  // Дальность
                point.r = colors[i][0];
                point.g = colors[i][1];
                point.b = colors[i][2];
                cloud->push_back(point);
                ++i;
        }
    }

    pcl::io::savePCDFileASCII("Coloroutput.pcd", *cloud);
    std::cout << "Range View saved to " << "Coloroutput.pcd" << std::endl;
}



  }  // namespace segmentation
}  // namespace rangenet



int main()
{

  rangenet::segmentation::Net test("../darknet53");
  test.getPoints("../cloud");
  test.doProjection_origin();
  //test.init_model();
  //test.convertToPointCloud(test.getLabels(test.infer(), 7200));

  return 0;
}