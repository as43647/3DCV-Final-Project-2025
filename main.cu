#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <algorithm>
#include <set>


// 定義資料結構
struct Point {
  float x = 0.0f, y = 0.0f, z = 0.0f;
  int r = 255, g = 255, b = 255;
  float point_size = 0.0f;
  int count = 0;
  int id = -1; // 追蹤頻寬的ID
};

struct Point_rgb {
  int r = 0, g = 0, b = 0;
  int count = 0;
  float depth_val = 100.0f; 
  int id = -1; 
};

struct Gaze {
  Point position;
  Point direction;
};

struct Vector3d {
  float x, y, z;
};

// 數學函式
__host__ __device__ void normalizeVector(Vector3d &v) {
    float magnitude = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (magnitude != 0) {
        v.x /= magnitude; v.y /= magnitude; v.z /= magnitude;
    }
}

__host__ __device__ void crossProduct(Vector3d A, Vector3d B, Vector3d &C) {
   C.x = A.y * B.z - A.z * B.y;
   C.y = -(A.x * B.z - A.z * B.x);
   C.z = A.x * B.y - A.y * B.x;
   normalizeVector(C);
}

// CUDA Kernels
__device__ void atomicMinn(float *const addr, const float val) {
  if (*addr <= val) return;
  unsigned int *const addr_as_ui = (unsigned int *)addr;
  unsigned int old = *addr_as_ui, assumed;
  do {
    assumed = old;
    if (__uint_as_float(assumed) <= val) break;
    old = atomicCAS(addr_as_ui, assumed, __float_as_uint(val));
  } while (assumed != old);
}

__global__ void setlogPolar(int numPoints, Point_rgb* logPolarBuffer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;
    logPolarBuffer[idx].depth_val = 100.0f;
    logPolarBuffer[idx].count = 0;
}

__global__ void logPolarTransformKernel(Point* points, int numPoints, float radius_min, int r_bins, int theta_bins, int phi_bins, Point_rgb* logPolarBuffer, Gaze gaze, Vector3d rightVector, Vector3d upVector, float rate_adapt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    Point point = points[idx];
    
    // 計算相對坐標
    float rel_x = point.x - gaze.position.x;
    float rel_y = point.y - gaze.position.y;
    float rel_z = point.z - gaze.position.z;

    // 投影到Gaze方向的深度(d)
    float d = rel_x * gaze.direction.x + rel_y * gaze.direction.y + rel_z * gaze.direction.z;
    if (d <= 0.001f) return; 

    // 投影到視平面
    float plane_x = rel_x / d;
    float plane_y = rel_y / d;
    float plane_z = rel_z / d;

    // 計算平面上的局部坐標
    float local_x = plane_x * rightVector.x + plane_y * rightVector.y + plane_z * rightVector.z;
    float local_y = plane_x * upVector.x + plane_y * upVector.y + plane_z * upVector.z;

    // 轉換為極座標
    float theta = atan2(local_y, local_x);
    float radius = max(sqrt(local_x*local_x + local_y*local_y), radius_min);

    // 計算Log-Polar索引
    float log_r = log(radius/radius_min);
    float base = (theta_bins + M_PI) / (theta_bins - M_PI);
    
    int r_index = (int)(log_r / log(base) / rate_adapt);
    int theta_index = (int)(theta_bins * (theta + M_PI) / (2 * M_PI));
    int phi_index = 0;

    if (r_index >= r_bins || theta_index >= theta_bins || r_index < 0) return;

    int buf_idx = r_index * theta_bins * phi_bins + theta_index * phi_bins + phi_index;

    // 更新最小深度
    atomicMinn(&logPolarBuffer[buf_idx].depth_val, d);
    
    // 更新顏色與ID
    if (d <= logPolarBuffer[buf_idx].depth_val + 0.002f){
        logPolarBuffer[buf_idx].r = point.r;
        logPolarBuffer[buf_idx].g = point.g;
        logPolarBuffer[buf_idx].b = point.b;
        logPolarBuffer[buf_idx].count = 1;
        logPolarBuffer[buf_idx].id = point.id; // 儲存ID
    }
}

__global__ void logPolarToCartesianKernel(Point_rgb* logPolarBuffer, int r_bins, int theta_bins, int phi_bins, Point* points, int max_save_points, float radius_min, Gaze gaze, Vector3d rightVector, Vector3d upVector, float rate_adapt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_save_points) return;

    if (logPolarBuffer[idx].depth_val >= 99.0f || logPolarBuffer[idx].count == 0) {
        points[idx].count = 0;
        return;
    }

    int r_index = idx / (theta_bins * phi_bins);
    int theta_index = (idx / phi_bins) % theta_bins;

    float base = (theta_bins + M_PI) / (theta_bins - M_PI);
    float r = radius_min * pow(base, (float)r_index * rate_adapt);
    float theta = (float) (2 * M_PI) * (float(theta_index) / float(theta_bins)) - M_PI;

    float x_back = r * cos(theta);
    float y_back = r * sin(theta);
    
    Vector3d new_dir;
    new_dir.x = gaze.direction.x + x_back * rightVector.x + y_back * upVector.x;
    new_dir.y = gaze.direction.y + x_back * rightVector.y + y_back * upVector.y;
    new_dir.z = gaze.direction.z + x_back * rightVector.z + y_back * upVector.z;

    float d = logPolarBuffer[idx].depth_val;

    points[idx].x = gaze.position.x + d * new_dir.x;
    points[idx].y = gaze.position.y + d * new_dir.y;
    points[idx].z = gaze.position.z + d * new_dir.z;
    
    points[idx].r = logPolarBuffer[idx].r;
    points[idx].g = logPolarBuffer[idx].g;
    points[idx].b = logPolarBuffer[idx].b;
    points[idx].id = logPolarBuffer[idx].id; // 取回 ID
    points[idx].count = 1;
}

// Host端I/O與前處理

std::vector<Point> loadPLY(const char* filename) {
    std::vector<Point> points;
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Cannot open file: %s\n", filename);
        return points;
    }

    char line[1024];
    int numPoints = 0;
    bool headerEnd = false;

    while (fgets(line, 1024, file)) {
        if (strncmp(line, "element vertex", 14) == 0) sscanf(line + 14, "%d", &numPoints);
        if (strncmp(line, "end_header", 10) == 0) { headerEnd = true; break; }
    }

    if (!headerEnd) return points;

    points.resize(numPoints);
    float min_y = 10000.0f, max_y = -10000.0f;
    float sum_x = 0, sum_y = 0, sum_z = 0;

    for (int i = 0; i < numPoints; i++) {
        if (!fgets(line, 1024, file)) break;
        int r=255, g=255, b=255;
        // 嘗試讀取6個值，如果失敗則給預設顏色
        int parsed = sscanf(line, "%f %f %f %d %d %d", &points[i].x, &points[i].y, &points[i].z, &r, &g, &b);
        if (parsed < 6) { r=200; g=200; b=200; }
        
        points[i].r = r; points[i].g = g; points[i].b = b;
        points[i].id = i; // 設定原始ID

        sum_x += points[i].x;
        sum_y += points[i].y;
        sum_z += points[i].z;
        if(points[i].y < min_y) min_y = points[i].y;
        if(points[i].y > max_y) max_y = points[i].y;
    }
    fclose(file);

    // 自動正規化
    float avg_x = sum_x / numPoints;
    float avg_y = sum_y / numPoints;
    float avg_z = sum_z / numPoints;
    float height = max_y - min_y;
    if (height == 0) height = 1.0f;

    float target_height = 1.8f; 
    float scale = target_height / height;
    
    printf("[Pre-process] Centering to (0,0,0) and Scaling by %.5f (Height: %.1f -> %.1f)\n", scale, height, target_height);

    for (int i = 0; i < numPoints; i++) {
        points[i].x = (points[i].x - avg_x) * scale;
        points[i].y = (points[i].y - avg_y) * scale;
        points[i].z = (points[i].z - avg_z) * scale;
    }
    return points;
}

void savePLY(const char* filename, const std::vector<Point>& points) {
    std::ofstream file(filename);
    int valid_count = 0;
    for(const auto& p : points) if(p.count > 0) valid_count++;

    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << valid_count << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "end_header\n";

    for (const auto& p : points) {
        if (p.count > 0) {
            file << p.x << " " << p.y << " " << p.z << " " 
                 << (int)p.r << " " << (int)p.g << " " << (int)p.b << "\n";
        }
    }
    file.close();
}

int main(int argc, char** argv) {
    const char* defaultPrefix = "loot_vox10_"; 
    const char* filePrefix = defaultPrefix;

    if (argc > 1) {
        filePrefix = argv[1];
    }

    int startFrame = 1000;      // 起始幀號
    int total_frames = 300;      // 要跑幾幀
    int max_capacity = 2000000; 
    
    Point* d_points;
    cudaMalloc(&d_points, max_capacity * sizeof(Point));

    // 參數設定
    float rate_adapt = 1.8f;
    // int r_bins = 600; int theta_bins = 450; int phi_bins = 1; // 稍微降低解析度以減少透視
    int r_bins = 1200; int theta_bins = 1500; int phi_bins = 1;
    float radius_min = 0.00008f; 
    int max_buffer_size = r_bins * theta_bins * phi_bins;

    Point_rgb* d_logPolarBuffer;
    cudaMalloc(&d_logPolarBuffer, max_buffer_size * sizeof(Point_rgb));
    Point* d_save_points;
    cudaMalloc(&d_save_points, max_buffer_size * sizeof(Point));
    std::vector<Point> result_points(max_buffer_size);

    int blockSize = 1024;
    int bufferBlocks = (max_buffer_size + blockSize - 1) / blockSize;

    // 相機設定
    float distance = 2.5f; 

    // 每一幀讀新檔+轉視角
    for (int i = 0; i < total_frames; i++) {
        
        
        char inputFilename[128];
        sprintf(inputFilename, "%s%d.ply", filePrefix, startFrame + i);
        
        // 讀取點雲
        printf("Loading Frame %d: %s...\n", i, inputFilename);
        std::vector<Point> host_points = loadPLY(inputFilename); 
        int numPoints = host_points.size();
        
        if (numPoints == 0) {
            printf("Error: File not found or empty!\n");
            continue;
        }

        
        cudaMemcpy(d_points, host_points.data(), numPoints * sizeof(Point), cudaMemcpyHostToDevice);
        int numBlocks = (numPoints + blockSize - 1) / blockSize;

        // 計算旋轉角度
        float angle = (float)i * (2.0f * M_PI / (float)total_frames);
        
        Gaze current_gaze;
        current_gaze.position.x = distance * sin(angle);
        current_gaze.position.y = 0.0f; 
        current_gaze.position.z = distance * cos(angle);

        current_gaze.direction.x = -current_gaze.position.x;
        current_gaze.direction.y = -current_gaze.position.y;
        current_gaze.direction.z = -current_gaze.position.z;
        
        Vector3d dir = {current_gaze.direction.x, current_gaze.direction.y, current_gaze.direction.z};
        normalizeVector(dir);
        current_gaze.direction.x = dir.x; current_gaze.direction.y = dir.y; current_gaze.direction.z = dir.z;

        Vector3d rightVec, upVec;
        crossProduct({current_gaze.direction.x, current_gaze.direction.y, current_gaze.direction.z}, {0,1,0}, rightVec);
        crossProduct(rightVec, {current_gaze.direction.x, current_gaze.direction.y, current_gaze.direction.z}, upVec);

        // 執行HPR Kernel
        setlogPolar<<<bufferBlocks, blockSize>>>(max_buffer_size, d_logPolarBuffer);
        logPolarTransformKernel<<<numBlocks, blockSize>>>(d_points, numPoints, radius_min, r_bins, theta_bins, phi_bins, d_logPolarBuffer, current_gaze, rightVec, upVec, rate_adapt);
        cudaDeviceSynchronize();
        
        // 重建點雲
        logPolarToCartesianKernel<<<bufferBlocks, blockSize>>>(d_logPolarBuffer, r_bins, theta_bins, phi_bins, d_save_points, max_buffer_size, radius_min, current_gaze, rightVec, upVec, rate_adapt);
        cudaDeviceSynchronize();

        // 取回結果
        cudaMemcpy(result_points.data(), d_save_points, max_buffer_size * sizeof(Point), cudaMemcpyDeviceToHost);

        // 計算HPR
        int visible_count = 0;
        for(const auto& p : result_points) {
            if(p.count > 0) visible_count++;
        }

        float saving = 100.0f * (1.0f - (float)visible_count / (float)numPoints);
        printf("  [Report] Original: %d -> HPR: %d | Saving: %.2f%%\n", numPoints, visible_count, saving);

        // 存檔
        char outputFilename[64];
        sprintf(outputFilename, "output/result_first_%02d.ply", i);
        savePLY(outputFilename, result_points);
    }

    cudaFree(d_points);
    cudaFree(d_logPolarBuffer);
    cudaFree(d_save_points);
    printf("Done!\n");
    return 0;
}
