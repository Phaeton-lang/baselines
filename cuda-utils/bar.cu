#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#define SIZE 1
using namespace std;

int main() {
  // freopen("swapinout.txt", "w", stdout);
  long int s[] = {
      //1,         4,          8,         16,        32,        40,
      //64,        68,         80,        128,       256,       400,
      //512,       768,        1024,      1535,      1536,      2048,
      //2264,      2348,       2600,      2888,      3020,      3164,
      //4096,      4716,       5120,      8192,      10240,     16384,
      //17920,     20000,      25600,     32768,
      37632,     42452,
      49152,     75276,      92928,     98304,     100000,    102400,
      131072,    147456,     204800,    262144,    294912,    307200,
      409860,    524288,     589824,    1000000,   1048576,   1179648,
      1228800,   1280000,    1581056,   1638400,   1638916,   2359296,
      2654208,   3538944,    4718592,   6422528,   8306688,   8388608,
      9437184,   10000000,   11075584,  11943936,  12845056,  16613376,
      16777216,  20447232,   25000000,  25690112,  35831808,  38535168,
      39321600,  49561600,   51380224,  67108864,  83886080,  121228800,
      142655492, 178438148,  186482692, 191102980, 196608000, 205520896,
      237568004, 1258291200, 3200000000};
  cout << "size,in,out" << endl;
  for (int i = 0; i < 59; i++) {
    long long size = s[i];
    void *hostArray = (void *)0;
    cudaMallocHost(&hostArray, size);
    void *deviceArray = (void *)0;
    cudaMalloc((void **)&deviceArray, size);

    long long a0 = std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
    // auto a0 = (std::chrono::system_clock::now()).time_since_epoch().count();
    cudaMemcpy(deviceArray, hostArray, size, cudaMemcpyHostToDevice);
    long long b0 = std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
    // auto b0 = (std::chrono::system_clock::now()).time_since_epoch().count();
    cudaMemcpy(hostArray, deviceArray, size, cudaMemcpyDeviceToHost);
    // long long c0 = std::chrono::duration_cast<std::chrono::nanoseconds>(
    //                   std::chrono::system_clock::now().time_since_epoch())
    //                   .count();
    auto c0 = (std::chrono::system_clock::now()).time_since_epoch().count();
    cout << s[i] << "," << b0 - a0 << "," << c0 - b0 << endl;
  }

  return 0;
}
