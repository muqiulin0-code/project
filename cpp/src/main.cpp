#include <charconv>
#include <cstddef>
#include <exception>
#include <iostream>
#include <string_view>

#include "producer_consumer/pipeline.hpp"

namespace {

std::size_t parse_size(std::string_view text, std::string_view name) {
  std::size_t value = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size()) {
    throw std::invalid_argument("invalid " + std::string(name) + ": " + std::string(text));
  }
  return value;
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    producer_consumer::PipelineOptions options;
    if (argc > 1) options.producer_count = parse_size(argv[1], "producer_count");
    if (argc > 2) options.consumer_count = parse_size(argv[2], "consumer_count");
    if (argc > 3) options.items_per_producer = parse_size(argv[3], "items_per_producer");
    if (argc > 4) options.queue_capacity = parse_size(argv[4], "queue_capacity");
    if (argc > 5) {
      throw std::invalid_argument(
          "usage: pc_demo [producers] [consumers] [items_per_producer] [queue_capacity]");
    }

    const auto stats = producer_consumer::run_pipeline(options);
    std::cout << "produced=" << stats.produced << " consumed=" << stats.consumed
              << " checksum=" << stats.checksum << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
