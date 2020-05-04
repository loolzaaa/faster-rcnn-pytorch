#include "nms.h"
#include "roi_pool.h"
#include "roi_align.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "Non-maximum suppression (CPU/CUDA)");
  m.def("roi_pool_forward", &roi_pool_forward, "ROI Polling forward pass (CPU/CUDA)");
  m.def("roi_pool_backward", &roi_pool_backward, "ROI Polling backward pass (CPU/CUDA)");
  m.def("roi_align_forward", &roi_align_forward, "ROI Align forward pass (CPU/CUDA)");
  m.def("roi_align_backward", &roi_align_backward, "ROI Align backward pass (CPU/CUDA)");
}