#pragma once

#include <cstddef>
#include <cstdint>

namespace producer_consumer {

struct PipelineOptions {
  std::size_t producer_count{2};
  std::size_t consumer_count{2};
  std::size_t items_per_producer{1000};
  std::size_t queue_capacity{32};
};

struct PipelineStats {
  std::size_t produced{0};
  std::size_t consumed{0};
  std::int64_t checksum{0};
};

// Run a complete producer-consumer workload and return deterministic statistics.
// Basic validation is performed before any thread is created.
PipelineStats run_pipeline(const PipelineOptions& options);

}  // namespace producer_consumer
