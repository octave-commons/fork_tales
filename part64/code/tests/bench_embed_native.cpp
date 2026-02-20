#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

struct Config {
  std::string model;
  std::string backend = "ort";
  std::string device = "AUTO";
  std::string verify_device = "warn";
  std::string inputs_path;
  std::string tokens_path;
  int iters = 200;
  int warmup = 20;
  int batch = 1;
  int dim = 0;
  int seq_len = 128;
  int threads = 1;
  int pin = 0;
  std::string priority = "normal";
  std::string power = "ignore";
  std::string timing = "model_only";
  std::string out = "runs/embed_bench";
  std::string tag = "default";
};

static std::string to_upper(std::string value) {
  for (char& c : value) {
    if (c >= 'a' && c <= 'z') {
      c = static_cast<char>(c - ('a' - 'A'));
    }
  }
  return value;
}

static std::string json_escape(const std::string& input) {
  std::string out;
  out.reserve(input.size() + 8);
  for (char c : input) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out.push_back(c);
        break;
    }
  }
  return out;
}

static int parse_int(const std::string& value, int fallback) {
  try {
    return std::stoi(value);
  } catch (...) {
    return fallback;
  }
}

static Config parse_args(int argc, char** argv) {
  Config cfg;

  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto next_value = [&](const std::string& default_value = "") -> std::string {
      if (i + 1 >= argc) {
        return default_value;
      }
      ++i;
      return argv[i];
    };

    if (key == "--model") {
      cfg.model = next_value();
    } else if (key == "--backend") {
      cfg.backend = next_value("ort");
    } else if (key == "--device") {
      cfg.device = to_upper(next_value("AUTO"));
    } else if (key == "--verify-device") {
      cfg.verify_device = next_value("warn");
    } else if (key == "--inputs") {
      cfg.inputs_path = next_value();
    } else if (key == "--tokens") {
      cfg.tokens_path = next_value();
    } else if (key == "--n") {
      cfg.iters = std::max(1, parse_int(next_value("200"), 200));
    } else if (key == "--warmup") {
      cfg.warmup = std::max(0, parse_int(next_value("20"), 20));
    } else if (key == "--batch") {
      cfg.batch = std::max(1, parse_int(next_value("1"), 1));
    } else if (key == "--dim") {
      cfg.dim = std::max(0, parse_int(next_value("0"), 0));
    } else if (key == "--seq-len") {
      cfg.seq_len = std::max(8, parse_int(next_value("128"), 128));
    } else if (key == "--threads") {
      cfg.threads = std::max(1, parse_int(next_value("1"), 1));
    } else if (key == "--pin") {
      cfg.pin = std::max(0, std::min(1, parse_int(next_value("0"), 0)));
    } else if (key == "--priority") {
      cfg.priority = next_value("normal");
    } else if (key == "--power") {
      cfg.power = next_value("ignore");
    } else if (key == "--timing") {
      cfg.timing = next_value("model_only");
    } else if (key == "--out") {
      cfg.out = next_value("runs/embed_bench");
    } else if (key == "--tag") {
      cfg.tag = next_value("default");
    }
  }

  return cfg;
}

static std::vector<std::string> read_lines(const std::string& path) {
  std::vector<std::string> lines;
  std::ifstream in(path);
  if (!in.is_open()) {
    return lines;
  }

  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty()) {
      lines.push_back(line);
    }
  }
  return lines;
}

static std::vector<int64_t> parse_token_line(const std::string& line, int seq_len, int* out_token_len) {
  std::vector<int64_t> out(static_cast<size_t>(seq_len), 0);
  int count = 0;
  std::string token;
  std::stringstream ss(line);

  while (std::getline(ss, token, ',')) {
    std::stringstream ws(token);
    std::string item;
    while (ws >> item) {
      if (count >= seq_len) {
        break;
      }
      out[static_cast<size_t>(count)] = static_cast<int64_t>(parse_int(item, 0));
      ++count;
    }
    if (count >= seq_len) {
      break;
    }
  }

  *out_token_len = count;
  return out;
}

static void tokenize_text(const std::string& text,
                          int seq_len,
                          int64_t* out_ids,
                          int64_t* out_mask,
                          int64_t* out_type,
                          int* out_token_len) {
  int count = 0;
  out_ids[0] = 101;
  out_mask[0] = 1;
  out_type[0] = 0;
  count = 1;

  for (char c : text) {
    if (count >= seq_len - 1) {
      break;
    }
    const unsigned char uc = static_cast<unsigned char>(c);
    out_ids[count] = static_cast<int64_t>((uc % 127) + 1000);
    out_mask[count] = 1;
    out_type[count] = 0;
    ++count;
  }

  if (count < seq_len) {
    out_ids[count] = 102;
    out_mask[count] = 1;
    out_type[count] = 0;
    ++count;
  }

  for (int i = count; i < seq_len; ++i) {
    out_ids[i] = 0;
    out_mask[i] = 0;
    out_type[i] = 0;
  }

  *out_token_len = count;
}

static double percentile_sorted(const std::vector<uint64_t>& sorted, double p) {
  if (sorted.empty()) {
    return 0.0;
  }
  if (sorted.size() == 1) {
    return static_cast<double>(sorted[0]);
  }

  const double pos = p * static_cast<double>(sorted.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = static_cast<size_t>(std::ceil(pos));
  if (lo == hi) {
    return static_cast<double>(sorted[lo]);
  }
  const double w = pos - static_cast<double>(lo);
  return static_cast<double>(sorted[lo]) * (1.0 - w) + static_cast<double>(sorted[hi]) * w;
}

static std::string read_first_cpu_model() {
  std::ifstream in("/proc/cpuinfo");
  if (!in.is_open()) {
    return "unknown";
  }
  std::string line;
  while (std::getline(in, line)) {
    const std::string prefix = "model name\t: ";
    if (line.rfind(prefix, 0) == 0) {
      return line.substr(prefix.size());
    }
  }
  return "unknown";
}

static std::string read_mem_total_mb() {
  std::ifstream in("/proc/meminfo");
  if (!in.is_open()) {
    return "unknown";
  }
  std::string line;
  while (std::getline(in, line)) {
    const std::string prefix = "MemTotal:";
    if (line.rfind(prefix, 0) == 0) {
      std::stringstream ss(line.substr(prefix.size()));
      long long kb = 0;
      ss >> kb;
      return std::to_string(kb / 1024LL);
    }
  }
  return "unknown";
}

int main(int argc, char** argv) {
  const Config cfg = parse_args(argc, argv);

  if (cfg.model.empty()) {
    std::cerr << "Missing --model <path>" << std::endl;
    return 4;
  }

  std::vector<std::string> providers_available;
  try {
    providers_available = Ort::GetAvailableProviders();
  } catch (...) {
    providers_available.clear();
  }

  std::cout << "Available providers:";
  for (const auto& p : providers_available) {
    std::cout << " " << p;
  }
  std::cout << std::endl;

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "embed_bench");
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(cfg.threads);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  std::string selected_device = "CPU";
  auto provider_exists = [&](const std::string& provider_name) {
    return std::find(providers_available.begin(), providers_available.end(), provider_name) !=
           providers_available.end();
  };

  try {
    if (cfg.device == "CUDA" || cfg.device == "GPU") {
      OrtCUDAProviderOptions cuda_opts;
      cuda_opts.device_id = 0;
      session_options.AppendExecutionProvider_CUDA(cuda_opts);
      selected_device = "CUDA";
    } else if (cfg.device == "NPU") {
      std::unordered_map<std::string, std::string> ov_opts;
      ov_opts["device_type"] = "NPU";
      session_options.AppendExecutionProvider_OpenVINO_V2(ov_opts);
      selected_device = "NPU";
    } else if (cfg.device == "AUTO") {
      if (provider_exists("OpenVINOExecutionProvider")) {
        std::unordered_map<std::string, std::string> ov_opts;
        ov_opts["device_type"] = "NPU";
        session_options.AppendExecutionProvider_OpenVINO_V2(ov_opts);
        selected_device = "NPU";
      } else if (provider_exists("CUDAExecutionProvider")) {
        OrtCUDAProviderOptions cuda_opts;
        cuda_opts.device_id = 0;
        session_options.AppendExecutionProvider_CUDA(cuda_opts);
        selected_device = "CUDA";
      } else {
        selected_device = "CPU";
      }
    } else {
      selected_device = "CPU";
    }
  } catch (const Ort::Exception& ex) {
    if (cfg.verify_device == "strict") {
      std::cerr << "Device verification failed while selecting provider: " << ex.what()
                << std::endl;
      return 2;
    }
    std::cerr << "Provider selection warning: " << ex.what() << std::endl;
    selected_device = "CPU";
  }

  if (cfg.verify_device == "strict" && cfg.device != "AUTO") {
    const std::string requested = to_upper(cfg.device);
    bool ok = (requested == "CPU" && selected_device == "CPU") ||
              ((requested == "CUDA" || requested == "GPU") && selected_device == "CUDA") ||
              (requested == "NPU" && selected_device == "NPU");
    if (!ok) {
      std::cerr << "Strict device verification failed. requested=" << requested
                << " selected=" << selected_device << std::endl;
      return 2;
    }
  }

  std::vector<std::string> raw_inputs;
  std::vector<std::vector<int64_t>> token_ids;
  std::vector<int> token_lens;

  if (!cfg.tokens_path.empty()) {
    const auto token_lines = read_lines(cfg.tokens_path);
    for (const auto& line : token_lines) {
      int len = 0;
      auto row = parse_token_line(line, cfg.seq_len, &len);
      token_ids.push_back(std::move(row));
      token_lens.push_back(len);
    }
  } else {
    if (!cfg.inputs_path.empty()) {
      raw_inputs = read_lines(cfg.inputs_path);
    }
    if (raw_inputs.empty()) {
      raw_inputs = {
          "hello world",
          "deterministic embeddings for simulation",
          "c runtime latency benchmark for npu",
          "matryoshka representation learning dimension slicing",
      };
    }

    // model_only: pre-tokenize once to isolate inference latency
    if (cfg.timing == "model_only") {
      for (const auto& text : raw_inputs) {
        std::vector<int64_t> row(static_cast<size_t>(cfg.seq_len), 0);
        std::vector<int64_t> mask(static_cast<size_t>(cfg.seq_len), 0);
        std::vector<int64_t> type(static_cast<size_t>(cfg.seq_len), 0);
        int len = 0;
        tokenize_text(text, cfg.seq_len, row.data(), mask.data(), type.data(), &len);
        token_ids.push_back(std::move(row));
        token_lens.push_back(len);
      }
    }
  }

  if (token_ids.empty() && raw_inputs.empty()) {
    std::cerr << "No valid inputs loaded" << std::endl;
    return 4;
  }

  const auto t_compile_start = std::chrono::steady_clock::now();
  Ort::Session session(nullptr);
  try {
    session = Ort::Session(env, cfg.model.c_str(), session_options);
  } catch (const Ort::Exception& ex) {
    std::cerr << "Session creation failed: " << ex.what() << std::endl;
    return 4;
  }
  const auto t_compile_end = std::chrono::steady_clock::now();
  const double compile_ms = std::chrono::duration<double, std::milli>(t_compile_end - t_compile_start).count();

  const int64_t batch = static_cast<int64_t>(cfg.batch);
  const int64_t seq_len = static_cast<int64_t>(cfg.seq_len);
  const size_t flat_size = static_cast<size_t>(batch * seq_len);
  const size_t dataset_count = std::max<size_t>(1, token_ids.empty() ? raw_inputs.size() : token_ids.size());

  std::vector<int64_t> input_ids(flat_size, 0);
  std::vector<int64_t> attention_mask(flat_size, 0);
  std::vector<int64_t> token_type_ids(flat_size, 0);
  std::vector<int64_t> position_ids(flat_size, 0);

  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<int64_t> input_shape = {batch, seq_len};
  Ort::AllocatorWithDefaultOptions allocator;

  const size_t model_input_count = session.GetInputCount();
  const size_t model_output_count = session.GetOutputCount();
  if (model_input_count == 0 || model_output_count == 0) {
    std::cerr << "Model has no inputs or outputs" << std::endl;
    return 3;
  }

  std::vector<std::string> model_input_names;
  model_input_names.reserve(model_input_count);
  for (size_t i = 0; i < model_input_count; ++i) {
    auto name = session.GetInputNameAllocated(i, allocator);
    model_input_names.emplace_back(name.get() ? name.get() : "");
  }

  auto output_name_alloc = session.GetOutputNameAllocated(0, allocator);
  const std::string output_name = output_name_alloc.get() ? output_name_alloc.get() : "";
  if (output_name.empty()) {
    std::cerr << "Output name is empty" << std::endl;
    return 3;
  }

  std::vector<std::vector<int64_t>> extra_input_buffers;
  std::vector<std::vector<float>> extra_float_buffers;
  std::vector<const char*> input_names;
  std::vector<Ort::Value> run_inputs;
  input_names.reserve(model_input_count);
  run_inputs.reserve(model_input_count);

  auto resolve_shape = [&](const std::string& input_name, const std::vector<int64_t>& raw_shape) {
    std::vector<int64_t> resolved = raw_shape;
    for (size_t d = 0; d < resolved.size(); ++d) {
      if (resolved[d] > 0 && resolved[d] <= 1000000) {
        continue;
      }
      if (d == 0) {
        resolved[d] = batch;
      } else if ((input_name.find("past_key_values") != std::string::npos ||
                  input_name.find("past.") != std::string::npos) &&
                 d == 2) {
        resolved[d] = 0;
      } else if (d == 1) {
        resolved[d] = seq_len;
      } else {
        resolved[d] = 1;
      }
    }
    return resolved;
  };

  const bool debug_shapes = (std::getenv("EMBED_BENCH_DEBUG") != nullptr);

  for (size_t input_index = 0; input_index < model_input_names.size(); ++input_index) {
    const auto& name = model_input_names[input_index];
    input_names.push_back(name.c_str());

    auto input_type_info = session.GetInputTypeInfo(input_index);
    if (input_type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
      std::cerr << "Unsupported non-tensor input: " << name << std::endl;
      return 3;
    }

    auto type_info = input_type_info.GetTensorTypeAndShapeInfo();
    const auto elem_type = type_info.GetElementType();
    const size_t dims_count = type_info.GetDimensionsCount();
    if (dims_count > 16) {
      std::cerr << "Input has unreasonable rank (" << dims_count << ") for " << name << std::endl;
      return 3;
    }

    std::vector<int64_t> raw_shape(dims_count, -1);
    if (dims_count > 0) {
      type_info.GetDimensions(raw_shape.data(), dims_count);
    }
    const auto expected_shape = resolve_shape(name, raw_shape);

    size_t element_count = 1;
    for (int64_t dim : expected_shape) {
      if (dim == 0) {
        element_count = 0;
        break;
      }
      element_count *= static_cast<size_t>(std::max<int64_t>(1, dim));
    }

    if (debug_shapes) {
      std::cout << "input=" << name << " elem_type=" << static_cast<int>(elem_type)
                << " elements=" << element_count << " shape=[";
      for (size_t i = 0; i < expected_shape.size(); ++i) {
        if (i > 0) {
          std::cout << ",";
        }
        std::cout << expected_shape[i];
      }
      std::cout << "]" << std::endl;
    }

    try {
      if (name == "input_ids") {
        run_inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
            mem_info,
            input_ids.data(),
            input_ids.size(),
            expected_shape.data(),
            expected_shape.size()));
      } else if (name == "attention_mask") {
        run_inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
            mem_info,
            attention_mask.data(),
            attention_mask.size(),
            expected_shape.data(),
            expected_shape.size()));
      } else if (name == "token_type_ids") {
        run_inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
            mem_info,
            token_type_ids.data(),
            token_type_ids.size(),
            expected_shape.data(),
            expected_shape.size()));
      } else if (name == "position_ids") {
        run_inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
            mem_info,
            position_ids.data(),
            position_ids.size(),
            expected_shape.data(),
            expected_shape.size()));
      } else {
        if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
          extra_input_buffers.emplace_back(std::max<size_t>(1, element_count), 0);
          run_inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
              mem_info,
              extra_input_buffers.back().data(),
              element_count,
              expected_shape.data(),
              expected_shape.size()));
        } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          extra_float_buffers.emplace_back(std::max<size_t>(1, element_count), 0.0f);
          run_inputs.emplace_back(Ort::Value::CreateTensor<float>(
              mem_info,
              extra_float_buffers.back().data(),
              element_count,
              expected_shape.data(),
              expected_shape.size()));
        } else {
          std::cerr << "Unsupported input type for " << name << " elem_type=" << elem_type
                    << std::endl;
          return 3;
        }
      }
    } catch (const std::exception& ex) {
      std::cerr << "Input tensor creation failed for " << name << ": " << ex.what() << std::endl;
      return 3;
    }
  }

  std::vector<const char*> output_names = {output_name.c_str()};

  auto fill_batch = [&](int iter_index) {
    for (int b = 0; b < cfg.batch; ++b) {
      const size_t row_offset = static_cast<size_t>(b) * static_cast<size_t>(cfg.seq_len);
      const size_t sample_index = static_cast<size_t>((iter_index * cfg.batch + b) % dataset_count);

      int current_len = 0;
      if (!token_ids.empty()) {
        const auto& sample = token_ids[sample_index % token_ids.size()];
        std::memcpy(&input_ids[row_offset], sample.data(), sizeof(int64_t) * static_cast<size_t>(cfg.seq_len));
        current_len = token_lens[sample_index % token_lens.size()];
      } else {
        tokenize_text(raw_inputs[sample_index % raw_inputs.size()],
                      cfg.seq_len,
                      &input_ids[row_offset],
                      &attention_mask[row_offset],
                      &token_type_ids[row_offset],
                      &current_len);
      }

      if (!token_ids.empty()) {
        for (int i = 0; i < cfg.seq_len; ++i) {
          attention_mask[row_offset + static_cast<size_t>(i)] = (i < current_len) ? 1 : 0;
          token_type_ids[row_offset + static_cast<size_t>(i)] = 0;
          position_ids[row_offset + static_cast<size_t>(i)] = static_cast<int64_t>(i);
        }
      } else {
        for (int i = 0; i < cfg.seq_len; ++i) {
          position_ids[row_offset + static_cast<size_t>(i)] = static_cast<int64_t>(i);
        }
      }
    }
  };

  fill_batch(0);

  uint64_t first_infer_us = 0;
  std::vector<Ort::Value> first_outputs;
  try {
    const auto t0 = std::chrono::steady_clock::now();
    first_outputs = session.Run(Ort::RunOptions{nullptr},
                                input_names.data(),
                                run_inputs.data(),
                                run_inputs.size(),
                                output_names.data(),
                                output_names.size());
    const auto t1 = std::chrono::steady_clock::now();
    first_infer_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
  } catch (const Ort::Exception& ex) {
    std::cerr << "First inference failed: " << ex.what() << std::endl;
    return 3;
  }

  if (first_outputs.empty() || !first_outputs[0].IsTensor()) {
    std::cerr << "Correctness failed: output is not tensor" << std::endl;
    return 3;
  }

  auto out_info = first_outputs[0].GetTensorTypeAndShapeInfo();
  auto out_shape = out_info.GetShape();
  if (out_shape.size() < 3) {
    std::cerr << "Correctness failed: unexpected output rank " << out_shape.size() << std::endl;
    return 3;
  }

  const int64_t hidden_dim = out_shape.back();
  if (hidden_dim <= 0) {
    std::cerr << "Correctness failed: hidden dim invalid" << std::endl;
    return 3;
  }

  const int used_dim = (cfg.dim <= 0) ? static_cast<int>(hidden_dim)
                                      : std::min(cfg.dim, static_cast<int>(hidden_dim));

  int64_t output_elements = 1;
  for (int64_t d : out_shape) {
    output_elements *= d;
  }
  std::vector<float> output_buffer(static_cast<size_t>(output_elements), 0.0f);
  Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
      mem_info, output_buffer.data(), output_buffer.size(), out_shape.data(), out_shape.size());

  Ort::IoBinding io_binding(session);
  for (size_t i = 0; i < input_names.size(); ++i) {
    io_binding.BindInput(input_names[i], run_inputs[i]);
  }
  io_binding.BindOutput(output_names[0], output_tensor);

  std::vector<float> pooled(static_cast<size_t>(cfg.batch) * static_cast<size_t>(used_dim), 0.0f);

  for (int i = 0; i < cfg.warmup; ++i) {
    fill_batch(i + 1);
    session.Run(Ort::RunOptions{nullptr}, io_binding);
  }

  std::vector<uint64_t> latencies_ns;
  latencies_ns.reserve(static_cast<size_t>(cfg.iters));
  std::vector<int> iter_token_lens;
  iter_token_lens.reserve(static_cast<size_t>(cfg.iters));

  for (int i = 0; i < cfg.iters; ++i) {
    int token_len_for_row0 = 0;
    const size_t sample_index = static_cast<size_t>(i %
                                                    std::max<size_t>(1, token_lens.empty() ? 1 : token_lens.size()));
    if (!token_lens.empty()) {
      token_len_for_row0 = token_lens[sample_index];
    }

    const auto t0 = std::chrono::steady_clock::now();

    if (cfg.timing == "boundary") {
      fill_batch(i);
    }

    session.Run(Ort::RunOptions{nullptr}, io_binding);

    if (cfg.timing == "boundary") {
      for (int b = 0; b < cfg.batch; ++b) {
        const size_t pooled_offset = static_cast<size_t>(b) * static_cast<size_t>(used_dim);
        std::fill(pooled.begin() + static_cast<long>(pooled_offset),
                  pooled.begin() + static_cast<long>(pooled_offset + static_cast<size_t>(used_dim)),
                  0.0f);

        int64_t valid = 0;
        for (int t = 0; t < cfg.seq_len; ++t) {
          const int64_t mask = attention_mask[static_cast<size_t>(b) * static_cast<size_t>(cfg.seq_len) +
                                              static_cast<size_t>(t)];
          if (mask <= 0) {
            continue;
          }
          ++valid;
          const size_t base = (static_cast<size_t>(b) * static_cast<size_t>(cfg.seq_len) +
                               static_cast<size_t>(t)) *
                              static_cast<size_t>(hidden_dim);
          for (int d = 0; d < used_dim; ++d) {
            pooled[pooled_offset + static_cast<size_t>(d)] += output_buffer[base + static_cast<size_t>(d)];
          }
        }

        if (valid > 0) {
          const float inv = 1.0f / static_cast<float>(valid);
          double norm_sq = 0.0;
          for (int d = 0; d < used_dim; ++d) {
            const float v = pooled[pooled_offset + static_cast<size_t>(d)] * inv;
            pooled[pooled_offset + static_cast<size_t>(d)] = v;
            norm_sq += static_cast<double>(v) * static_cast<double>(v);
          }
          const double norm = std::sqrt(norm_sq);
          if (norm > 1e-12) {
            const float inv_norm = static_cast<float>(1.0 / norm);
            for (int d = 0; d < used_dim; ++d) {
              pooled[pooled_offset + static_cast<size_t>(d)] *= inv_norm;
            }
          }
        }
      }
    }

    const auto t1 = std::chrono::steady_clock::now();
    latencies_ns.push_back(static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()));
    iter_token_lens.push_back(token_len_for_row0);
  }

  if (latencies_ns.empty()) {
    std::cerr << "No latency samples collected" << std::endl;
    return 4;
  }

  std::vector<uint64_t> sorted = latencies_ns;
  std::sort(sorted.begin(), sorted.end());
  const double mean_ns = static_cast<double>(
      std::accumulate(latencies_ns.begin(), latencies_ns.end(), 0.0) / static_cast<double>(latencies_ns.size()));

  double var_ns = 0.0;
  for (uint64_t x : latencies_ns) {
    const double dx = static_cast<double>(x) - mean_ns;
    var_ns += dx * dx;
  }
  var_ns /= static_cast<double>(latencies_ns.size());
  const double stddev_ns = std::sqrt(var_ns);

  const double p50_ns = percentile_sorted(sorted, 0.50);
  const double p90_ns = percentile_sorted(sorted, 0.90);
  const double p99_ns = percentile_sorted(sorted, 0.99);
  const double max_ns = static_cast<double>(sorted.back());

  const double spike_threshold_ns = p99_ns;
  size_t spikes = 0;
  for (uint64_t x : latencies_ns) {
    if (static_cast<double>(x) >= spike_threshold_ns) {
      ++spikes;
    }
  }
  const double spike_rate = static_cast<double>(spikes) / static_cast<double>(latencies_ns.size());

  int min_len = cfg.seq_len;
  int max_len = 0;
  long long sum_len = 0;
  if (!iter_token_lens.empty()) {
    for (int len : iter_token_lens) {
      min_len = std::min(min_len, len);
      max_len = std::max(max_len, len);
      sum_len += len;
    }
  }

  fs::create_directories(cfg.out);
  const fs::path run_json_path = fs::path(cfg.out) / "run.json";
  const fs::path latency_csv_path = fs::path(cfg.out) / "latency.csv";

  {
    std::ofstream csv(latency_csv_path);
    csv << "iter,input_index,tokens_len,latency_ns\n";
    for (size_t i = 0; i < latencies_ns.size(); ++i) {
      const size_t input_index = i % dataset_count;
      const int tokens_len = iter_token_lens.empty() ? 0 : iter_token_lens[i];
      csv << i << "," << input_index << "," << tokens_len << "," << latencies_ns[i] << "\n";
    }
  }

  {
    std::ofstream json(run_json_path);
    json << "{\n";
    json << "  \"tag\": \"" << json_escape(cfg.tag) << "\",\n";
    json << "  \"backend\": \"" << json_escape(cfg.backend) << "\",\n";
    json << "  \"device_requested\": \"" << json_escape(cfg.device) << "\",\n";
    json << "  \"device_selected\": \"" << json_escape(selected_device) << "\",\n";
    json << "  \"verify_device\": \"" << json_escape(cfg.verify_device) << "\",\n";
    json << "  \"ort_version\": \"" << json_escape(Ort::GetVersionString()) << "\",\n";
    json << "  \"cpu_model\": \"" << json_escape(read_first_cpu_model()) << "\",\n";
    json << "  \"mem_total_mb\": \"" << json_escape(read_mem_total_mb()) << "\",\n";
    json << "  \"power_mode\": \"" << json_escape(cfg.power) << "\",\n";
    json << "  \"timing\": \"" << json_escape(cfg.timing) << "\",\n";
    json << "  \"model\": \"" << json_escape(cfg.model) << "\",\n";
    json << "  \"batch\": " << cfg.batch << ",\n";
    json << "  \"seq_len\": " << cfg.seq_len << ",\n";
    json << "  \"dim\": " << used_dim << ",\n";
    json << "  \"threads\": " << cfg.threads << ",\n";
    json << "  \"pin\": " << cfg.pin << ",\n";
    json << "  \"providers_available\": [";
    for (size_t i = 0; i < providers_available.size(); ++i) {
      if (i > 0) {
        json << ",";
      }
      json << "\"" << json_escape(providers_available[i]) << "\"";
    }
    json << "],\n";
    json << "  \"dataset\": {\n";
    json << "    \"count\": " << latencies_ns.size() << ",\n";
    json << "    \"token_len_min\": " << min_len << ",\n";
    json << "    \"token_len_mean\": "
         << (iter_token_lens.empty() ? 0.0
                                     : static_cast<double>(sum_len) / static_cast<double>(iter_token_lens.size()))
         << ",\n";
    json << "    \"token_len_max\": " << max_len << "\n";
    json << "  },\n";
    json << "  \"summary\": {\n";
    json << "    \"compile_ms\": " << compile_ms << ",\n";
    json << "    \"first_infer_us\": " << first_infer_us << ",\n";
    json << "    \"p50_us\": " << (p50_ns / 1000.0) << ",\n";
    json << "    \"p90_us\": " << (p90_ns / 1000.0) << ",\n";
    json << "    \"p99_us\": " << (p99_ns / 1000.0) << ",\n";
    json << "    \"max_us\": " << (max_ns / 1000.0) << ",\n";
    json << "    \"mean_us\": " << (mean_ns / 1000.0) << ",\n";
    json << "    \"stddev_us\": " << (stddev_ns / 1000.0) << ",\n";
    json << "    \"spike_rate\": " << spike_rate << "\n";
    json << "  }\n";
    json << "}\n";
  }

  std::cout << "Selected device: " << selected_device << std::endl;
  std::cout << "compile_ms=" << compile_ms << " first_infer_us=" << first_infer_us
            << " p99_us=" << (p99_ns / 1000.0) << " mean_us=" << (mean_ns / 1000.0)
            << " max_us=" << (max_ns / 1000.0) << std::endl;
  std::cout << "Artifacts: " << run_json_path << " and " << latency_csv_path << std::endl;

  return 0;
}
