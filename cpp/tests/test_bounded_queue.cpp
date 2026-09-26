#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <thread>

#include "producer_consumer/bounded_queue.hpp"
#include "producer_consumer/pipeline.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void test_fifo_order() {
  producer_consumer::BoundedQueue<int> queue(2);
  require(queue.push(10), "push 10 failed");
  require(queue.push(20), "push 20 failed");
  int value = 0;
  require(queue.pop(value) && value == 10, "FIFO order failed for first item");
  require(queue.pop(value) && value == 20, "FIFO order failed for second item");
}

void test_capacity_blocks_until_pop() {
  producer_consumer::BoundedQueue<int> queue(1);
  require(queue.push(1), "initial push failed");
  std::atomic<bool> push_finished{false};
  std::thread producer([&] {
    queue.push(2);
    push_finished.store(true, std::memory_order_release);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  require(!push_finished.load(std::memory_order_acquire), "push did not block on full queue");
  int value = 0;
  require(queue.pop(value) && value == 1, "pop failed");
  producer.join();
  require(push_finished.load(std::memory_order_acquire), "blocked producer was not released");
}

void test_close_wakes_consumer() {
  producer_consumer::BoundedQueue<int> queue(1);
  std::atomic<bool> popped{true};
  std::thread consumer([&] {
    int value = 0;
    popped.store(queue.pop(value), std::memory_order_release);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  queue.close();
  consumer.join();
  require(!popped.load(std::memory_order_acquire), "closed empty queue returned a value");
}

void test_pipeline_workload() {
  producer_consumer::PipelineOptions options;
  options.producer_count = 4;
  options.consumer_count = 3;
  options.items_per_producer = 500;
  options.queue_capacity = 8;
  const auto stats = producer_consumer::run_pipeline(options);
  require(stats.produced == 2000, "unexpected produced count");
  require(stats.consumed == 2000, "unexpected consumed count");
  require(stats.checksum == 1999000, "unexpected checksum");
}

}  // namespace

int main() {
  try {
    test_fifo_order();
    test_capacity_blocks_until_pop();
    test_close_wakes_consumer();
    test_pipeline_workload();
    std::cout << "all bounded queue and pipeline tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return 1;
  }
}
