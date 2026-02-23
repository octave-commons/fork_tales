#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

std::mutex g_create_error_lock;
std::string g_last_create_error;

static void set_last_create_error(const std::string& value) {
  std::lock_guard<std::mutex> lock(g_create_error_lock);
  g_last_create_error = value;
}

static const char* get_last_create_error_cstr() {
  std::lock_guard<std::mutex> lock(g_create_error_lock);
  return g_last_create_error.c_str();
}

static std::string to_upper(std::string value) {
  for (char& c : value) {
    if (c >= 'a' && c <= 'z') {
      c = static_cast<char>(c - ('a' - 'A'));
    }
  }
  return value;
}

static std::string to_lower(std::string value) {
  for (char& c : value) {
    if (c >= 'A' && c <= 'Z') {
      c = static_cast<char>(c + ('a' - 'A'));
    }
  }
  return value;
}

static bool wants_npu_device(const std::string& value) {
  const std::string probe = to_upper(value);
  return probe == "NPU" || probe == "INTEL_NPU";
}

static bool wants_gpu_device(const std::string& value) {
  const std::string probe = to_upper(value);
  return probe == "GPU" || probe == "CUDA" || probe == "NVIDIA" ||
         probe == "NVIDIA_GPU";
}

static bool is_hardware_device(const std::string& value) {
  const std::string probe = to_upper(value);
  return probe == "NPU" || probe == "CUDA" || probe == "GPU";
}

static bool env_true(const char* key) {
  if (key == nullptr || std::strlen(key) == 0) {
    return false;
  }
  const char* raw = std::getenv(key);
  if (raw == nullptr) {
    return false;
  }
  const std::string probe = to_lower(std::string(raw));
  return probe == "1" || probe == "true" || probe == "yes" || probe == "on";
}

struct InputMeta {
  std::string name;
  ONNXTensorElementDataType elem_type;
  std::vector<int64_t> shape;
  size_t element_count;
};

struct CEmbedRuntime;

static bool message_has_cpu_fallback_signal(const std::string& message) {
  const std::string lower = to_lower(message);
  if (lower.empty()) {
    return false;
  }

  const bool has_cpu = lower.find("cpu") != std::string::npos;
  const bool has_fallback =
      (lower.find("fallback") != std::string::npos) ||
      (lower.find("fall back") != std::string::npos) ||
      (lower.find("falling back") != std::string::npos);
  if (has_cpu && has_fallback) {
    return true;
  }

  if (lower.find("ov cpu") != std::string::npos && has_fallback) {
    return true;
  }

  if (lower.find("ze_result_error_unsupported_feature") != std::string::npos && has_cpu) {
    return true;
  }

  if (lower.find("unsupported") != std::string::npos &&
      lower.find("npu") != std::string::npos && has_cpu) {
    return true;
  }

  return false;
}

static void ort_log_capture(void* param,
                            OrtLoggingLevel /*severity*/,
                            const char* /*category*/,
                            const char* /*logid*/,
                            const char* /*code_location*/,
                            const char* message);

struct CEmbedRuntime {
  Ort::Env env;
  Ort::Session session;
  Ort::MemoryInfo mem_info;
  std::vector<InputMeta> inputs;
  std::vector<const char*> input_name_ptrs;
  std::string output_name;
  int64_t seq_len;
  int out_dim;
  std::string selected_device;
  std::string last_error;
  bool deny_cpu_fallback;
  bool cpu_fallback_detected;
  std::string cpu_fallback_detail;
  std::mutex run_lock;
  std::mutex status_lock;

  // Pooled buffers to avoid per-call allocation.
  std::vector<int64_t> pooled_default_mask;
  std::vector<int64_t> pooled_default_type;
  std::vector<int64_t> pooled_position_ids;
  std::vector<std::vector<int64_t>> pooled_extra_i64;
  std::vector<std::vector<float>> pooled_extra_f32;
  std::vector<float> pooled_hidden;
  size_t run_inputs_capacity;

  CEmbedRuntime(const char* model_path,
                const char* requested_device,
                int64_t seq,
                int output_dim,
                int threads,
                bool strict)
      : env(ORT_LOGGING_LEVEL_WARNING, "c_embed_runtime", ort_log_capture, this),
        session(nullptr),
        mem_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
        seq_len(std::max<int64_t>(8, seq)),
        out_dim(std::max(1, output_dim)),
        selected_device("UNSET"),
        deny_cpu_fallback(strict),
        cpu_fallback_detected(false) {
    if (model_path == nullptr || std::strlen(model_path) == 0) {
      throw std::runtime_error("empty model path");
    }

    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(std::max(1, threads));
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    const std::string wanted_raw =
        to_upper(std::string(requested_device ? requested_device : "AUTO"));
    const std::string wanted = wanted_raw.empty() ? "AUTO" : wanted_raw;

    auto providers = Ort::GetAvailableProviders();
    auto provider_exists = [&](const std::string& provider_name) {
      return std::find(providers.begin(), providers.end(), provider_name) != providers.end();
    };

    std::vector<std::string> provider_failures;
    auto push_provider_failure = [&](const std::string& detail) {
      if (!detail.empty()) {
        provider_failures.push_back(detail);
      }
    };

    auto append_npu_provider = [&]() {
      if (!provider_exists("OpenVINOExecutionProvider")) {
        push_provider_failure("OpenVINOExecutionProvider unavailable");
        return false;
      }
      try {
        std::unordered_map<std::string, std::string> ov_opts;
        ov_opts["device_type"] = "NPU";
        ov_opts["disable_dynamic_shapes"] = "True";
        ov_opts["enable_qdq_optimizer"] = "True";
        options.AppendExecutionProvider_OpenVINO_V2(ov_opts);
        selected_device = "NPU";
        return true;
      } catch (const std::exception& ex) {
        push_provider_failure(std::string("OpenVINOExecutionProvider error:") + ex.what());
        return false;
      }
    };

    auto append_cuda_provider = [&]() {
      if (!provider_exists("CUDAExecutionProvider")) {
        push_provider_failure("CUDAExecutionProvider unavailable");
        return false;
      }
      try {
        OrtCUDAProviderOptions cuda_opts{};
        cuda_opts.device_id = 0;
        options.AppendExecutionProvider_CUDA(cuda_opts);
        selected_device = "CUDA";
        return true;
      } catch (const std::exception& ex) {
        push_provider_failure(std::string("CUDAExecutionProvider error:") + ex.what());
        return false;
      }
    };

    auto append_openvino_gpu_provider = [&]() {
      if (!env_true("CDB_EMBED_GPU_ALLOW_OPENVINO_FALLBACK")) {
        push_provider_failure("OpenVINO GPU fallback disabled");
        return false;
      }
      if (!provider_exists("OpenVINOExecutionProvider")) {
        push_provider_failure("OpenVINOExecutionProvider unavailable for GPU");
        return false;
      }
      try {
        std::unordered_map<std::string, std::string> ov_opts;
        ov_opts["device_type"] = "GPU";
        ov_opts["disable_dynamic_shapes"] = "True";
        options.AppendExecutionProvider_OpenVINO_V2(ov_opts);
        selected_device = "GPU";
        return true;
      } catch (const std::exception& ex) {
        push_provider_failure(std::string("OpenVINOExecutionProvider GPU error:") + ex.what());
        return false;
      }
    };

    bool provider_selected = false;
    if (wanted == "AUTO") {
      provider_selected = append_npu_provider();
      if (!provider_selected) {
        provider_selected = append_cuda_provider();
      }
      if (!provider_selected) {
        provider_selected = append_openvino_gpu_provider();
      }
    } else if (wants_npu_device(wanted)) {
      provider_selected = append_npu_provider();
    } else if (wants_gpu_device(wanted)) {
      provider_selected = append_cuda_provider();
      if (!provider_selected) {
        provider_selected = append_openvino_gpu_provider();
      }
    } else {
      throw std::runtime_error("unsupported requested_device (expected AUTO|NPU|GPU|CUDA)");
    }

    if (!provider_selected) {
      std::string detail;
      for (const auto& row : provider_failures) {
        if (row.empty()) {
          continue;
        }
        if (!detail.empty()) {
          detail += " | ";
        }
        detail += row;
      }
      if (detail.empty()) {
        detail = "no hardware execution provider could be selected";
      }
      throw std::runtime_error(detail);
    }

    session = Ort::Session(env, model_path, options);

    if (!is_hardware_device(selected_device)) {
      throw std::runtime_error("invalid selected device for embedding runtime");
    }

    Ort::AllocatorWithDefaultOptions allocator;
    const size_t input_count = session.GetInputCount();
    if (input_count == 0) {
      throw std::runtime_error("model has no inputs");
    }

    for (size_t i = 0; i < input_count; ++i) {
      auto name_alloc = session.GetInputNameAllocated(i, allocator);
      std::string name = name_alloc.get() ? name_alloc.get() : "";
      if (name.empty()) {
        throw std::runtime_error("empty input name");
      }

      auto type_info = session.GetInputTypeInfo(i);
      if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
        throw std::runtime_error("non-tensor input is unsupported");
      }

      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      const auto elem_type = tensor_info.GetElementType();
      const size_t dims_count = tensor_info.GetDimensionsCount();
      std::vector<int64_t> raw_shape(dims_count, -1);
      if (dims_count > 0) {
        tensor_info.GetDimensions(raw_shape.data(), dims_count);
      }

      std::vector<int64_t> shape = raw_shape;
      for (size_t d = 0; d < shape.size(); ++d) {
        int64_t value = shape[d];
        if (value > 0 && value < 1000000) {
          continue;
        }
        if (d == 0) {
          shape[d] = 1;
        } else if (d == 1) {
          shape[d] = seq_len;
        } else {
          shape[d] = 1;
        }
      }

      size_t element_count = 1;
      for (int64_t dim : shape) {
        if (dim <= 0) {
          dim = 1;
        }
        element_count *= static_cast<size_t>(dim);
      }

      inputs.push_back(InputMeta{name, elem_type, shape, element_count});
    }

    const size_t output_count = session.GetOutputCount();
    if (output_count == 0) {
      throw std::runtime_error("model has no outputs");
    }
    auto output_name_alloc = session.GetOutputNameAllocated(0, allocator);
    output_name = output_name_alloc.get() ? output_name_alloc.get() : "";
    if (output_name.empty()) {
      throw std::runtime_error("empty output name");
    }

    input_name_ptrs.reserve(inputs.size());
    for (const auto& meta : inputs) {
      input_name_ptrs.push_back(meta.name.c_str());
    }

    // Initialize pooled buffers to avoid per-call allocation.
    pooled_default_mask.assign(static_cast<size_t>(seq_len), 1);
    pooled_default_type.assign(static_cast<size_t>(seq_len), 0);
    pooled_position_ids.resize(static_cast<size_t>(seq_len));
    for (int64_t i = 0; i < seq_len; ++i) {
      pooled_position_ids[static_cast<size_t>(i)] = i;
    }

    // Pre-allocate extra input buffers based on model input types.
    size_t extra_i64_count = 0;
    size_t extra_f32_count = 0;
    for (const auto& meta : inputs) {
      if (meta.name == "input_ids" || meta.name == "attention_mask" ||
          meta.name == "token_type_ids" || meta.name == "position_ids") {
        continue;
      }
      if (meta.elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        ++extra_i64_count;
      } else if (meta.elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        ++extra_f32_count;
      }
    }
    pooled_extra_i64.resize(extra_i64_count);
    pooled_extra_f32.resize(extra_f32_count);
    run_inputs_capacity = inputs.size();

    // Pre-allocate hidden buffer to max expected dimension.
    pooled_hidden.assign(2048, 0.0f);
  }
};

static void ort_log_capture(void* param,
                            OrtLoggingLevel /*severity*/,
                            const char* /*category*/,
                            const char* /*logid*/,
                            const char* /*code_location*/,
                            const char* message) {
  auto* runtime = static_cast<CEmbedRuntime*>(param);
  if (runtime == nullptr || message == nullptr) {
    return;
  }
  const std::string text(message);
  if (!message_has_cpu_fallback_signal(text)) {
    return;
  }

  std::lock_guard<std::mutex> status_lock(runtime->status_lock);
  runtime->cpu_fallback_detected = true;
  if (runtime->cpu_fallback_detail.empty()) {
    runtime->cpu_fallback_detail = text.substr(0, 640);
  }
}

static void fold_and_normalize(const float* pooled,
                               int hidden,
                               float* out,
                               int out_dim) {
  std::fill(out, out + out_dim, 0.0f);

  if (hidden <= 0) {
    return;
  }

  if (hidden >= out_dim) {
    for (int i = 0; i < out_dim; ++i) {
      out[i] = pooled[i];
    }
  } else {
    for (int i = 0; i < hidden; ++i) {
      out[i % out_dim] += pooled[i];
    }
  }

  double norm_sq = 0.0;
  for (int i = 0; i < out_dim; ++i) {
    norm_sq += static_cast<double>(out[i]) * static_cast<double>(out[i]);
  }
  if (norm_sq <= 1e-12) {
    return;
  }

  const float inv = static_cast<float>(1.0 / std::sqrt(norm_sq));
  for (int i = 0; i < out_dim; ++i) {
    out[i] *= inv;
  }
}

static int run_embed(CEmbedRuntime* runtime,
                     const int64_t* input_ids,
                     const int64_t* attention_mask,
                     const int64_t* token_type_ids,
                     float* out_vec) {
  if (runtime == nullptr || input_ids == nullptr || out_vec == nullptr) {
    return 0;
  }

  // Use pooled buffers instead of per-call allocation.
  int64_t* mask_ptr = const_cast<int64_t*>(attention_mask);
  if (mask_ptr == nullptr) {
    mask_ptr = runtime->pooled_default_mask.data();
  }
  int64_t* type_ptr = const_cast<int64_t*>(token_type_ids);
  if (type_ptr == nullptr) {
    type_ptr = runtime->pooled_default_type.data();
  }

  int64_t* position_ids_ptr = runtime->pooled_position_ids.data();

  // Reset and reuse extra input buffers.
  size_t extra_i64_idx = 0;
  size_t extra_f32_idx = 0;
  for (const auto& meta : runtime->inputs) {
    if (meta.name == "input_ids" || meta.name == "attention_mask" ||
        meta.name == "token_type_ids" || meta.name == "position_ids") {
      continue;
    }
    if (meta.elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      if (extra_i64_idx < runtime->pooled_extra_i64.size()) {
        auto& buf = runtime->pooled_extra_i64[extra_i64_idx];
        if (buf.size() != meta.element_count) {
          buf.assign(meta.element_count, 0);
        } else {
          std::fill(buf.begin(), buf.end(), 0);
        }
        ++extra_i64_idx;
      }
    } else if (meta.elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      if (extra_f32_idx < runtime->pooled_extra_f32.size()) {
        auto& buf = runtime->pooled_extra_f32[extra_f32_idx];
        if (buf.size() != meta.element_count) {
          buf.assign(meta.element_count, 0.0f);
        } else {
          std::fill(buf.begin(), buf.end(), 0.0f);
        }
        ++extra_f32_idx;
      }
    }
  }

  std::vector<Ort::Value> run_inputs;
  run_inputs.reserve(runtime->run_inputs_capacity);

  {
    std::lock_guard<std::mutex> status_lock(runtime->status_lock);
    if (runtime->deny_cpu_fallback && runtime->cpu_fallback_detected) {
      runtime->last_error = runtime->cpu_fallback_detail.empty()
                                ? "npu_cpu_fallback_detected"
                                : std::string("npu_cpu_fallback_detected:") +
                                      runtime->cpu_fallback_detail;
      return 0;
    }
  }

  try {
    for (const auto& meta : runtime->inputs) {
      const auto& shape = meta.shape;
      if (meta.name == "input_ids") {
        run_inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
            runtime->mem_info,
            const_cast<int64_t*>(input_ids),
            static_cast<size_t>(runtime->seq_len),
            shape.data(),
            shape.size()));
      } else if (meta.name == "attention_mask") {
        run_inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
            runtime->mem_info,
            const_cast<int64_t*>(mask_ptr),
            static_cast<size_t>(runtime->seq_len),
            shape.data(),
            shape.size()));
      } else if (meta.name == "token_type_ids") {
        run_inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
            runtime->mem_info,
            const_cast<int64_t*>(type_ptr),
            static_cast<size_t>(runtime->seq_len),
            shape.data(),
            shape.size()));
      } else if (meta.name == "position_ids") {
        run_inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
            runtime->mem_info,
            position_ids_ptr,
            static_cast<size_t>(runtime->seq_len),
            shape.data(),
            shape.size()));
      } else if (meta.elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        // Use pooled buffer for extra int64 inputs.
        if (extra_i64_idx < runtime->pooled_extra_i64.size()) {
          auto& buf = runtime->pooled_extra_i64[extra_i64_idx];
          run_inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
              runtime->mem_info,
              buf.data(),
              buf.size(),
              shape.data(),
              shape.size()));
          ++extra_i64_idx;
        }
      } else if (meta.elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        // Use pooled buffer for extra float inputs.
        if (extra_f32_idx < runtime->pooled_extra_f32.size()) {
          auto& buf = runtime->pooled_extra_f32[extra_f32_idx];
          run_inputs.emplace_back(Ort::Value::CreateTensor<float>(
              runtime->mem_info,
              buf.data(),
              buf.size(),
              shape.data(),
              shape.size()));
          ++extra_f32_idx;
        }
      } else {
        runtime->last_error = "unsupported_input_type";
        return 0;
      }
    }

    const char* output_name = runtime->output_name.c_str();
    auto outputs = runtime->session.Run(Ort::RunOptions{nullptr},
                                        runtime->input_name_ptrs.data(),
                                        run_inputs.data(),
                                        run_inputs.size(),
                                        &output_name,
                                        1);

    {
      std::lock_guard<std::mutex> status_lock(runtime->status_lock);
      if (runtime->deny_cpu_fallback && runtime->cpu_fallback_detected) {
        runtime->last_error = runtime->cpu_fallback_detail.empty()
                                  ? "npu_cpu_fallback_detected"
                                  : std::string("npu_cpu_fallback_detected:") +
                                        runtime->cpu_fallback_detail;
        return 0;
      }
    }

    if (outputs.empty() || !outputs[0].IsTensor()) {
      runtime->last_error = "output_not_tensor";
      return 0;
    }

    auto out_info = outputs[0].GetTensorTypeAndShapeInfo();
    auto out_shape = out_info.GetShape();
    const float* out_data = outputs[0].GetTensorData<float>();
    if (out_data == nullptr || out_shape.empty()) {
      runtime->last_error = "empty_output";
      return 0;
    }

    int64_t hidden = out_shape.back();
    if (hidden <= 0 || hidden > 16384) {
      runtime->last_error = "invalid_hidden_dim";
      return 0;
    }

    const int hidden_dim = static_cast<int>(hidden);
    // Use pooled hidden buffer, resize if needed.
    if (runtime->pooled_hidden.size() < static_cast<size_t>(hidden_dim)) {
      runtime->pooled_hidden.assign(static_cast<size_t>(hidden_dim), 0.0f);
    } else {
      std::fill(runtime->pooled_hidden.begin(),
                runtime->pooled_hidden.begin() + hidden_dim, 0.0f);
    }
    float* pooled = runtime->pooled_hidden.data();

    if (out_shape.size() >= 3) {
      int64_t token_dim = out_shape[out_shape.size() - 2];
      if (token_dim <= 0) {
        runtime->last_error = "invalid_token_dim";
        return 0;
      }

      const int64_t token_count = std::min<int64_t>(runtime->seq_len, token_dim);
      int64_t valid = 0;
      for (int64_t t = 0; t < token_count; ++t) {
        if (mask_ptr[t] <= 0) {
          continue;
        }
        ++valid;
        const size_t base = static_cast<size_t>(t) * static_cast<size_t>(hidden_dim);
        for (int d = 0; d < hidden_dim; ++d) {
          pooled[static_cast<size_t>(d)] += out_data[base + static_cast<size_t>(d)];
        }
      }

      if (valid <= 0) {
        valid = 1;
        for (int d = 0; d < hidden_dim; ++d) {
          pooled[static_cast<size_t>(d)] = out_data[static_cast<size_t>(d)];
        }
      } else {
        const float inv = 1.0f / static_cast<float>(valid);
        for (int d = 0; d < hidden_dim; ++d) {
          pooled[static_cast<size_t>(d)] *= inv;
        }
      }
    } else if (out_shape.size() == 2) {
      for (int d = 0; d < hidden_dim; ++d) {
        pooled[static_cast<size_t>(d)] = out_data[static_cast<size_t>(d)];
      }
    } else {
      runtime->last_error = "unsupported_output_rank";
      return 0;
    }

    fold_and_normalize(pooled, hidden_dim, out_vec, runtime->out_dim);
    runtime->last_error.clear();
    return 1;
  } catch (const Ort::Exception& ex) {
    runtime->last_error = std::string("ort_error:") + ex.what();
    return 0;
  } catch (const std::exception& ex) {
    runtime->last_error = std::string("std_error:") + ex.what();
    return 0;
  } catch (...) {
    runtime->last_error = "unknown_error";
    return 0;
  }
}

}  // namespace

extern "C" {

void* c_embed_runtime_create(const char* model_path,
                             const char* requested_device,
                             int64_t seq_len,
                             int32_t out_dim,
                             int32_t threads,
                             int32_t strict) {
  try {
    auto* runtime = new CEmbedRuntime(model_path,
                                      requested_device,
                                      seq_len,
                                      out_dim,
                                      threads,
                                      strict != 0);
    set_last_create_error("");
    return runtime;
  } catch (const Ort::Exception& ex) {
    set_last_create_error(std::string("ort_create_error:") + ex.what());
    return nullptr;
  } catch (const std::exception& ex) {
    set_last_create_error(std::string("std_create_error:") + ex.what());
    return nullptr;
  } catch (...) {
    set_last_create_error("unknown_create_error");
    return nullptr;
  }
}

const char* c_embed_runtime_last_create_error() {
  return get_last_create_error_cstr();
}

int32_t c_embed_runtime_embed(void* handle,
                              const int64_t* input_ids,
                              const int64_t* attention_mask,
                              const int64_t* token_type_ids,
                              float* out_vec) {
  auto* runtime = static_cast<CEmbedRuntime*>(handle);
  if (runtime == nullptr) {
    return 0;
  }
  std::lock_guard<std::mutex> lock(runtime->run_lock);
  return run_embed(runtime, input_ids, attention_mask, token_type_ids, out_vec) ? 1 : 0;
}

const char* c_embed_runtime_last_error(void* handle) {
  auto* runtime = static_cast<CEmbedRuntime*>(handle);
  if (runtime == nullptr) {
    return "invalid_handle";
  }
  return runtime->last_error.c_str();
}

const char* c_embed_runtime_selected_device(void* handle) {
  auto* runtime = static_cast<CEmbedRuntime*>(handle);
  if (runtime == nullptr) {
    return "invalid";
  }
  return runtime->selected_device.c_str();
}

int32_t c_embed_runtime_cpu_fallback_detected(void* handle) {
  auto* runtime = static_cast<CEmbedRuntime*>(handle);
  if (runtime == nullptr) {
    return 0;
  }
  std::lock_guard<std::mutex> status_lock(runtime->status_lock);
  return runtime->cpu_fallback_detected ? 1 : 0;
}

const char* c_embed_runtime_cpu_fallback_detail(void* handle) {
  static thread_local std::string detail_copy;
  auto* runtime = static_cast<CEmbedRuntime*>(handle);
  if (runtime == nullptr) {
    detail_copy = "invalid_handle";
    return detail_copy.c_str();
  }
  std::lock_guard<std::mutex> status_lock(runtime->status_lock);
  detail_copy = runtime->cpu_fallback_detail;
  return detail_copy.c_str();
}

int32_t c_embed_runtime_output_dim(void* handle) {
  auto* runtime = static_cast<CEmbedRuntime*>(handle);
  if (runtime == nullptr) {
    return 0;
  }
  return runtime->out_dim;
}

int64_t c_embed_runtime_seq_len(void* handle) {
  auto* runtime = static_cast<CEmbedRuntime*>(handle);
  if (runtime == nullptr) {
    return 0;
  }
  return runtime->seq_len;
}

void c_embed_runtime_destroy(void* handle) {
  auto* runtime = static_cast<CEmbedRuntime*>(handle);
  delete runtime;
}

}  // extern "C"
