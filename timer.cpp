#pragma once

#include <iostream>
#include <chrono>

class Timer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    bool is_running = false;

public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
        is_running = true;
    }

    void stop(std::string mess="") {
        if (!is_running) {
            std::cerr << "Timer was not started." << std::endl;
            return;
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto start = std::chrono::time_point_cast<std::chrono::microseconds>(start_time).time_since_epoch().count();
        auto end = std::chrono::time_point_cast<std::chrono::microseconds>(end_time).time_since_epoch().count();

        auto duration = end - start;
        double ms = duration * 0.001;

        // std::cout << "[" << mess << "] duration: " << duration << "us (" << ms << "ms)" << std::endl;
        std::cout << "[" << mess << "] duration: " << ms << "ms" << std::endl;
        is_running = false;
    }
};

#ifdef TEST
int main() {
    Timer timer[2];
    timer[0].start();  // 手动开始计时
    
    for (int i = 0; i < 1000000; i++) {
        // 这只是一个示例循环
    }
    
    timer[0].stop("part1");  // 手动结束计时并输出结果

    return 0;
}
#endif