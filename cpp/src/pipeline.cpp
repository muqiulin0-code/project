#include "producer_consumer/pipeline.hpp"

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

#include "producer_consumer/bounded_queue.hpp"

namespace producer_consumer {
namespace {

std::int64_t expected_checksum(const PipelineOptions& options) {
  const auto items = static_cast<std::int64_t>(options.items_per_producer);
  const auto producers = static_cast<std::int64_t>(options.producer_count);
  const auto per_producer = items * (items - 1) / 2;
  return producers * per_producer +
         items * (producers * (producers - 1) / 2) * items;
}

}  // namespace

PipelineStats run_pipeline(const PipelineOptions& options) {
  if (options.producer_count == 0 || options.consumer_count == 0) {
    throw std::invalid_argument("producer_count and consumer_count must be greater than zero");
  }
  if (options.items_per_producer == 0) {
    throw std::invalid_argument("items_per_producer must be greater than zero");
  }

  BoundedQueue<std::int64_t> queue(options.queue_capacity);
  std::atomic<std::size_t> produced{0};
  std::vector<std::size_t> consumed_by_thread(options.consumer_count, 0);
  std::vector<std::int64_t> sum_by_thread(options.consumer_count, 0);
  std::vector<std::thread> producers;
  std::vector<std::thread> consumers;
  producers.reserve(options.producer_count);
  consumers.reserve(options.consumer_count);

  for (std::size_t producer = 0; producer < options.producer_count; ++producer) {
    producers.emplace_back([&, producer] {
      const auto base = static_cast<std::int64_t>(producer * options.items_per_producer);
      for (std::size_t item = 0; item < options.items_per_producer; ++item) {
        if (!queue.push(base + static_cast<std::int64_t>(item))) {
          return;
        }
        produced.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  for (std::size_t consumer = 0; consumer < options.consumer_count; ++consumer) {
    consumers.emplace_back([&, consumer] {
      std::int64_t value = 0;
      std::int64_t local_sum = 0;
      std::size_t local_count = 0;
      while (queue.pop(value)) {
        // Placeholder for real work. Summing verifies every value was delivered.
        local_sum += value;
        ++local_count;
      }
      sum_by_thread[consumer] = local_sum;
      consumed_by_thread[consumer] = local_count;
    });
  }

  for (auto& producer : producers) {
    producer.join();
  }
  queue.close();
  for (auto& consumer : consumers) {
    consumer.join();
  }

  std::int64_t checksum = 0;
  std::size_t consumed = 0;
  for (std::size_t consumer = 0; consumer < options.consumer_count; ++consumer) {
    checksum += sum_by_thread[consumer];
    consumed += consumed_by_thread[consumer];
  }

  const std::size_t actual_produced = produced.load(std::memory_order_relaxed);
  if (actual_produced != consumed || checksum != expected_checksum(options)) {
    throw std::runtime_error("producer-consumer pipeline lost or corrupted items");
  }
  return {actual_produced, consumed, checksum};
}

}  // namespace producer_consumer
