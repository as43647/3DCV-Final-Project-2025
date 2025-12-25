#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <limits>

// ============================================================================
// THEIA POINT CLOUD PROCESSING WITH ELLIPTICAL LOG-POLAR TRANSFORMATION
// ============================================================================
// 
// This program processes PLY point cloud files using gaze-based foveated
// rendering with elliptical log-polar transformation.
// 
// Key Features:
// 1. Occlusion-aware point culling
// 2. Elliptical log-polar transformation (horizontal vs vertical asymmetry)
// 3. Foveated streaming based on human visual system
// 4. Real-time GPU processing with CUDA
// 
// Elliptical Parameters:
//   a = 1.6 (horizontal semi-axis) -> more compression in horizontal direction
//   b = 1.0 (vertical semi-axis)   -> less compression in vertical direction
//   This matches human visual system where vertical resolution is more important
// 
// USAGE:
//   ./3dcv_final_elliptical [input_file] [output_file] [gaze_pos_x] [gaze_pos_y] [gaze_pos_z] 
//               [gaze_dir_x] [gaze_dir_y] [gaze_dir_z]
// 
// ============================================================================

typedef unsigned char BYTE;

// Structure for a 3D point with position and color
struct Point {
    float x = 0.0f, y = 0.0f, z = 0.0f;
    int r = 0, g = 0, b = 0;
    float point_size = 0.0f;
    int count = 0;
};

// Structure for log-polar buffer elements
struct Point_rgb {
    int r = 0, g = 0, b = 0;
    int count = 0;
    float depth_val = 1.0f;
};

// Structure for gaze position and direction
struct Gaze {
    Point position;
    Point direction;
};

// 3D vector structure
struct Vector3d {
    float x, y, z;
};

// Structure for ID and size
struct Id_Size {
    int id;
    float size;
};

// Elliptical parameters for log-polar transformation
// a > b means more compression in horizontal direction
const float ELLIPTICAL_A = 1.6f;  // Horizontal semi-axis
const float ELLIPTICAL_B = 1.0f;  // Vertical semi-axis

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Write log messages to file
void writeToLogFile_gpu(const std::string& logMessage) {
    std::ofstream logFile("../../Logs/log.txt", std::ios_base::app);
    if (logFile.is_open()) {
        logFile << logMessage << std::endl;
        logFile.close();
    } else {
        std::cerr << "Unable to open log file." << std::endl;
    }
}

// Get current time in milliseconds
uint64_t timeSinceEpochMillisec() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

uint64_t NDKGetTime() {
    return timeSinceEpochMillisec();
}

// ============================================================================
// CUDA KERNELS
// ============================================================================

// Custom atomic minimum for float values
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

// Kernel to calculate depth of points along gaze direction
__global__ void calculate_depth(Point* points, Gaze gaze, float* depth_list, int numPoints) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;
    
    Point point = points[idx];
    
    // Calculate relative position
    Point point_relative;
    point_relative.x = point.x - gaze.position.x;
    point_relative.y = point.y - gaze.position.y;
    point_relative.z = point.z - gaze.position.z;
    
    // Calculate depth along gaze direction (dot product)
    float d = point_relative.x * gaze.direction.x + 
              point_relative.y * gaze.direction.y + 
              point_relative.z * gaze.direction.z;
    
    depth_list[idx] = d;
}

// Kernel for elliptical log-polar transformation
__global__ void logPolarTransformKernel(Point* points, int numPoints, float radius_min, 
                                        int r_bins, int theta_bins, int phi_bins, 
                                        Point_rgb* logPolarBuffer, Gaze gaze, 
                                        Vector3d rightVector, Vector3d upVector, float rate_adapt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;
    
    Point point = points[idx];
    
    // Calculate relative position
    Point point_relative;
    point_relative.x = point.x - gaze.position.x;
    point_relative.y = point.y - gaze.position.y;
    point_relative.z = point.z - gaze.position.z;
    
    // Calculate depth along gaze direction
    float d = point_relative.x * gaze.direction.x + 
              point_relative.y * gaze.direction.y + 
              point_relative.z * gaze.direction.z;
    
    // Project point onto plane (perspective division)
    Point point_on_plane;
    point_on_plane.x = point_relative.x / d;
    point_on_plane.y = point_relative.y / d;
    point_on_plane.z = point_relative.z / d;
    
    // Project onto right and up vectors
    float x = point_on_plane.x * rightVector.x + 
              point_on_plane.y * rightVector.y + 
              point_on_plane.z * rightVector.z;
    
    float y = point_on_plane.x * upVector.x + 
              point_on_plane.y * upVector.y + 
              point_on_plane.z * upVector.z;
    
    // ======================================================================
    // ELLIPTICAL LOG-POLAR TRANSFORMATION
    // ======================================================================
    // Apply elliptical normalization (different compression in x vs y)
    float x_ellip = x / ELLIPTICAL_A;  // More compression in horizontal
    float y_ellip = y / ELLIPTICAL_B;  // Less compression in vertical
    
    // Calculate elliptical polar coordinates
    float theta = atan2(y_ellip, x_ellip);
    float radius = max(sqrt(x_ellip * x_ellip + y_ellip * y_ellip), radius_min);
    
    // Apply logarithmic scaling to radius
    float log_r = log(radius / radius_min);
    float base = (theta_bins + M_PI) / (theta_bins - M_PI);
    
    // Determine buffer indices
    int r_index = (int)(log_r / log(base) / rate_adapt);
    int theta_index = (int)(theta_bins * (theta + M_PI) / (2 * M_PI));
    int phi_index = 0;  // Single depth layer
    
    // Check if indices are within range
    if (r_index >= r_bins || theta_index >= theta_bins || phi_index >= phi_bins) return;
    
    // Calculate buffer position
    int tmp_position = r_index * theta_bins * phi_bins + theta_index * phi_bins + phi_index;
    
    // Atomic updates to avoid race conditions
    atomicAdd(&logPolarBuffer[tmp_position].r, point.r);
    atomicAdd(&logPolarBuffer[tmp_position].g, point.g);
    atomicAdd(&logPolarBuffer[tmp_position].b, point.b);
    atomicAdd(&logPolarBuffer[tmp_position].count, 1);
    atomicMinn(&logPolarBuffer[tmp_position].depth_val, d);
    
    // Debug output for first few points
    if (idx < 5 && threadIdx.x == 0) {
        printf("[GPU] Thread %d: Added color (%d,%d,%d) to cell %d\n", 
               idx, point.r, point.g, point.b, tmp_position);
    }
}

// Kernel to initialize log-polar buffer
__global__ void setlogPolar(int numPoints, Point_rgb* logPolarBuffer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;
    logPolarBuffer[idx].depth_val = 100.0f;  // Mark as empty
}

// Kernel for inverse elliptical log-polar transformation (back to Cartesian)
__global__ void logPolarToCartesianKernel(Point_rgb* logPolarBuffer, int r_bins, int theta_bins, 
                                         int phi_bins, Point* points, int numPoints, 
                                         float radius_min, float d_min, float d_max, 
                                         Gaze gaze, Vector3d rightVector, Vector3d upVector, 
                                         float rate_adapt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;
    
    // Compute indices in log-polar buffer
    int r_index = idx / (theta_bins * phi_bins);
    int theta_index = (idx / phi_bins) % theta_bins;
    int phi_index = idx % phi_bins;
    
    // Skip empty cells
    if (logPolarBuffer[idx].depth_val == 100.0f) {
        points[idx].x = points[idx].y = points[idx].z = 0.0f;
        points[idx].r = points[idx].g = points[idx].b = 0;
        points[idx].count = 0;
        return;
    }
    
    // Convert log-polar coordinates back to elliptical polar
    float base = (theta_bins + M_PI) / (theta_bins - M_PI);
    float rho_e = radius_min * std::pow(base, (float)r_index * rate_adapt);
    float theta_e = (float)(2 * M_PI) * (float(theta_index) / float(theta_bins)) - M_PI;
    
    // ======================================================================
    // INVERSE ELLIPTICAL TRANSFORMATION
    // ======================================================================
    // Convert elliptical polar to elliptical Cartesian
    float x_ellip_back = rho_e * cos(theta_e);
    float y_ellip_back = rho_e * sin(theta_e);
    
    // Convert elliptical Cartesian to original Cartesian (inverse normalization)
    float x_back = x_ellip_back * ELLIPTICAL_A;
    float y_back = y_ellip_back * ELLIPTICAL_B;
    
    // Reconstruct 3D direction
    Vector3d new_dir;
    new_dir.x = gaze.direction.x + x_back * rightVector.x + y_back * upVector.x;
    new_dir.y = gaze.direction.y + x_back * rightVector.y + y_back * upVector.y;
    new_dir.z = gaze.direction.z + x_back * rightVector.z + y_back * upVector.z;
    
    // Get depth value
    float d_back = logPolarBuffer[idx].depth_val;
    
    // Convert back to 3D Cartesian coordinates
    Point cartesian_point;
    cartesian_point.x = gaze.position.x + d_back * new_dir.x;
    cartesian_point.y = gaze.position.y + d_back * new_dir.y;
    cartesian_point.z = gaze.position.z + d_back * new_dir.z;
    
    // Compute average color of points in the cell
    int count = logPolarBuffer[idx].count;
    int red = (count > 0) ? (logPolarBuffer[idx].r / count) : 0;
    int green = (count > 0) ? (logPolarBuffer[idx].g / count) : 0;
    int blue = (count > 0) ? (logPolarBuffer[idx].b / count) : 0;
    
    // Calculate point size based on radial resolution
    float prev_rho = (r_index > 0) ? 
        radius_min * std::pow(base, (float)(r_index - 1) * rate_adapt) : radius_min;
    float point_size = d_back * (rho_e - prev_rho);
    
    // Store the reconstructed point
    points[idx].x = cartesian_point.x;
    points[idx].y = cartesian_point.y;
    points[idx].z = cartesian_point.z;
    points[idx].r = red;
    points[idx].g = green;
    points[idx].b = blue;
    points[idx].point_size = point_size;
    points[idx].count = 1;
}

// Kernel for filling empty buffers using nearest neighbor interpolation
__global__ void fill_empty_buffers_kernel(Point_rgb *temp_log_polar_buffer, 
                                          Point_rgb *temp_log_polar_buffer_filled, 
                                          int r_bins, int theta_bins, int phi_bins, 
                                          float radius_min, float base, float point_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_size = r_bins * theta_bins * phi_bins;
    
    if (idx >= total_size) return;
    
    // Calculate buffer indices
    int r = idx / (theta_bins * phi_bins);
    int theta = (idx / phi_bins) % theta_bins;
    int phi = idx % phi_bins;
    
    Point_rgb *current_point = &temp_log_polar_buffer_filled[r * theta_bins * phi_bins + theta * phi_bins + phi];
    Point_rgb *current_buffer_point = &temp_log_polar_buffer[r * theta_bins * phi_bins + theta * phi_bins + phi];
    
    // If cell is not empty, copy directly
    if (current_buffer_point->depth_val != 100.0f) {
        *current_point = *current_buffer_point;
        return;
    }
    
    // Search for nearest non-empty cell
    Point_rgb *nearest_point = nullptr;
    int min_distance = INT_MAX;
    
    // Calculate search ranges based on point size
    int r_diff = 0;
    for (int temp_r_diff = 0; temp_r_diff < r_bins - r; temp_r_diff++) {
        float temp_distance = radius_min * std::pow(base, (float)(temp_r_diff + r)) - 
                             radius_min * std::pow(base, (float)r);
        if (temp_distance > 1.25 * point_size) {
            r_diff = temp_r_diff;
            break;
        }
    }
    
    float temp_theta_distance = 2 * M_PI * radius_min * std::pow(base, (float)r) / float(theta_bins);
    int theta_diff = int(1.25 * point_size / temp_theta_distance);
    
    if (r_diff <= 1) r_diff = 0;
    if (theta_diff <= 1) theta_diff = 0;
    
    // Search within calculated ranges
    for (int r2 = min(r_bins, r + r_diff); r2 >= max(0, r - r_diff); r2--) {
        for (int theta2 = max(0, theta - theta_diff); theta2 < min(theta_bins, theta + theta_diff); ++theta2) {
            for (int phi2 = 0; phi2 < phi_bins; ++phi2) {
                if (r2 < r && nearest_point == nullptr) {
                    return;
                }
                
                Point_rgb *candidate_point = &temp_log_polar_buffer[r2 * theta_bins * phi_bins + theta2 * phi_bins + phi2];
                if (candidate_point->depth_val != 100.0f) {
                    int dist = abs(r - r2) + abs(theta - theta2) + abs(phi - phi2);
                    if (dist < min_distance) {
                        min_distance = dist;
                        nearest_point = candidate_point;
                    }
                }
            }
        }
    }
    
    // Copy from nearest point if found
    if (nearest_point != nullptr) {
        *current_point = *nearest_point;
    } else {
        current_point->r = current_point->g = current_point->b = 0;
        current_point->depth_val = 100.0f;
        current_point->count = 0;
    }
}

// Kernel to process raw point cloud data
__global__ void processPointsKernel(BYTE *pp_pc, int nPoints, Point *points) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < nPoints) {
        // Position data offset
        int pDataOffset = 15 + index * sizeof(short) * 3;
        // Color data offset
        int cDataOffset = 15 + nPoints * sizeof(short) * 3 + index * sizeof(char) * 3;
        
        // Read position
        short pp[3];
        memcpy(pp, pp_pc + pDataOffset, sizeof(short) * 3);
        points[index].x = (float)pp[0] / 1000.0f;
        points[index].y = (float)pp[1] / 1000.0f;
        points[index].z = -(float)pp[2] / 1000.0f;
        
        // Read color
        char pc[3];
        memcpy(pc, pp_pc + cDataOffset, sizeof(char) * 3);
        points[index].r = (int)pc[0];
        points[index].g = (int)pc[1];
        points[index].b = (int)pc[2];
    }
}

// ============================================================================
// HOST UTILITY FUNCTIONS
// ============================================================================

// Normalize a 3D vector
void normalizeVector(Vector3d &v) {
    float magnitude = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (magnitude != 0) {
        v.x /= magnitude;
        v.y /= magnitude;
        v.z /= magnitude;
    }
}

// Cross product of two vectors
void crossProduct(Vector3d A, Vector3d B, Vector3d &C) {
    C.x = A.y * B.z - A.z * B.y;
    C.y = -(A.x * B.z - A.z * B.x);
    C.z = A.x * B.y - A.y * B.x;
    normalizeVector(C);
}

// Read PLY file (ASCII format)
int readPLY(const char* filename, Point* points, int maxPoints) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        return -1;
    }
    
    // Read header
    char line[1024];
    int numPoints = 0;
    while (fgets(line, 1024, file)) {
        if (strncmp(line, "element vertex", 14) == 0) {
            sscanf(line + 14, "%d", &numPoints);
            if (numPoints > maxPoints) {
                numPoints = maxPoints;
            }
            break;
        }
    }
    
    // Skip to end of header
    while (fgets(line, 1024, file)) {
        if (strncmp(line, "end_header", 10) == 0) {
            break;
        }
    }
    
    // Read points
    for (int i = 0; i < numPoints; i++) {
        if (!fgets(line, 1024, file)) {
            break;
        }
        sscanf(line, "%f %f %f %d %d %d", 
               &points[i].x, &points[i].y, &points[i].z, 
               &points[i].r, &points[i].g, &points[i].b);
        
        // Apply transformations (specific to dataset)
        points[i].x = points[i].x * 0.181731f - 39.1599f;
        points[i].y = points[i].y * 0.181731f + 3.75652f;
        points[i].z = points[i].z * 0.181731f - 46.6228f;
        points[i].x /= 100.0f;
        points[i].y /= 100.0f;
        points[i].z /= 100.0f;
    }
    
    fclose(file);
    return numPoints;
}

// ============================================================================
// MAIN PROCESSING FUNCTION
// ============================================================================

extern "C"
void get_fovea(BYTE *pp_pc, int dataLen, Gaze gaze, Point* d_points, 
               std::vector<Point> &ret_points_log_polar_inner, 
               std::vector<Point> &ret_points_log_polar_outer, 
               bool dynamic_skip, float rate_adapt, bool aug) {
    
    uint64_t t0 = NDKGetTime();
    std::ostringstream oss;
    
    // Calculate number of points
    int nPoints = (dataLen - 15) / (sizeof(short) + sizeof(char)) / 3;
    
    // Allocate GPU memory for input data
    BYTE *d_pp_pc;
    cudaMalloc((void**)&d_pp_pc, dataLen * sizeof(BYTE));
    cudaMemcpy(d_pp_pc, pp_pc, dataLen * sizeof(BYTE), cudaMemcpyHostToDevice);
    
    // Process points on GPU
    int blockSize = 1024;
    int gridSize = (nPoints + blockSize - 1) / blockSize;
    processPointsKernel<<<gridSize, blockSize>>>(d_pp_pc, nPoints, d_points);
    
    uint64_t t1 = NDKGetTime();
    oss << "GPU read data time used: " << t1 - t0 << " ms" << std::endl;
    writeToLogFile_gpu(oss.str());
    oss.str("");
    
    // Log-polar parameters
    const int r_bins = (int)(400 / rate_adapt);
    const int theta_bins = (int)(285 / rate_adapt);
    const int phi_bins = 1;
    float point_size = 0.0018f;
    int max_points_save = r_bins * theta_bins * phi_bins;
    
    // Allocate host buffers
    Point_rgb* temp_log_polar_buffer = (Point_rgb*)malloc(sizeof(Point_rgb) * max_points_save);
    Point* save_points = (Point*)malloc(max_points_save * sizeof(Point));
    
    // Calculate right and up vectors
    Vector3d rightVector;
    crossProduct(Vector3d{gaze.direction.x, gaze.direction.y, gaze.direction.z}, 
                 Vector3d{0, 1, 0}, rightVector);
    
    Vector3d upVector;
    crossProduct(rightVector, 
                 Vector3d{gaze.direction.x, gaze.direction.y, gaze.direction.z}, 
                 upVector);
    
    // Copy vectors to constant memory
    cudaMemcpyToSymbol(rightVector, &rightVector, sizeof(Vector3d));
    cudaMemcpyToSymbol(upVector, &upVector, sizeof(Vector3d));
    
    float radius_min = 0.00029088821f;
    
    // Allocate and initialize GPU log-polar buffer
    Point_rgb* logPolarBuffer;
    cudaMalloc((void**)&logPolarBuffer, sizeof(Point_rgb) * max_points_save);
    
    Point* d_save_points;
    cudaMalloc(&d_save_points, max_points_save * sizeof(Point));
    
    // Initialize buffer
    setlogPolar<<<gridSize, blockSize>>>(max_points_save, logPolarBuffer);
    
    // Perform elliptical log-polar transformation
    logPolarTransformKernel<<<gridSize, blockSize>>>(d_points, nPoints, radius_min, 
                                                     r_bins, theta_bins, phi_bins, 
                                                     logPolarBuffer, gaze, 
                                                     rightVector, upVector, rate_adapt);
    
    // Copy results to host
    cudaMemcpy(temp_log_polar_buffer, logPolarBuffer, 
               sizeof(Point_rgb) * max_points_save, cudaMemcpyDeviceToHost);
    
    // Calculate depth statistics
    float d_min = std::numeric_limits<float>::max();
    float d_max = std::numeric_limits<float>::min();
    float depth_sum = 0;
    int depth_point_size = 0;
    
    for (int i = 0; i < max_points_save; i++) {
        float d = temp_log_polar_buffer[i].depth_val;
        if (d == 100.0f) continue;
        
        d_min = std::min(d_min, d);
        d_max = std::max(d_max, d);
        depth_sum += d;
        depth_point_size++;
    }
    
    float depth_mean = depth_sum / float(depth_point_size);
    float unit_point_size = point_size / d_min;
    float base = (theta_bins + M_PI) / (theta_bins - M_PI);
    
    // Allocate GPU buffers for inpainting
    Point_rgb *d_temp_log_polar_buffer;
    Point_rgb *d_temp_log_polar_buffer_filled;
    size_t buffer_size = max_points_save * sizeof(Point_rgb);
    
    cudaMalloc((void**)&d_temp_log_polar_buffer, buffer_size);
    cudaMalloc((void**)&d_temp_log_polar_buffer_filled, buffer_size);
    cudaMemcpy(d_temp_log_polar_buffer, temp_log_polar_buffer, 
               buffer_size, cudaMemcpyHostToDevice);
    
    // Perform inpainting if augmentation is enabled
    if (aug) {
        fill_empty_buffers_kernel<<<gridSize, blockSize>>>(d_temp_log_polar_buffer, 
                                                          d_temp_log_polar_buffer_filled, 
                                                          r_bins, theta_bins, phi_bins, 
                                                          radius_min, base, unit_point_size);
        
        // Inverse transformation with inpainted buffer
        logPolarToCartesianKernel<<<(max_points_save + blockSize - 1) / blockSize, blockSize>>>(
            d_temp_log_polar_buffer_filled, r_bins, theta_bins, phi_bins, 
            d_save_points, max_points_save, radius_min, d_min, d_max, 
            gaze, rightVector, upVector, rate_adapt);
    } else {
        // Inverse transformation without inpainting
        logPolarToCartesianKernel<<<(max_points_save + blockSize - 1) / blockSize, blockSize>>>(
            d_temp_log_polar_buffer, r_bins, theta_bins, phi_bins, 
            d_save_points, max_points_save, radius_min, d_min, d_max, 
            gaze, rightVector, upVector, rate_adapt);
    }
    
    // Copy results back to host
    cudaMemcpy(save_points, d_save_points, 
               max_points_save * sizeof(Point), cudaMemcpyDeviceToHost);
    
    uint64_t t5 = NDKGetTime();
    oss << "GPU Total time used before push_back: " << t5 - t0 << " ms" << std::endl;
    writeToLogFile_gpu(oss.str());
    oss.str("");
    
    // Separate points into inner (foveal) and outer regions
    int inner_point_idx = (int)(log(tan(0.15 * M_PI / 180.0) / radius_min) / 
                                log(base) / rate_adapt) * theta_bins;
    
    for (int i = 0; i < max_points_save; i++) {
        Point point = save_points[i];
        if (point.count == 0) continue;
        
        // Adjust point size
        float temp_point_size = std::min(0.018f * rate_adapt, 
                                        1.0f * point.point_size + 0.0007f);
        
        Point temp_point{point.x, point.y, point.z, 
                        point.r, point.g, point.b, 
                        temp_point_size};
        
        if (dynamic_skip) {
            if (i < inner_point_idx) {
                ret_points_log_polar_inner.push_back(temp_point);
            }
        } else {
            ret_points_log_polar_inner.push_back(temp_point);
        }
    }
    
    uint64_t t6 = NDKGetTime();
    oss << "GPU Total time used in GPU: " << t6 - t0 << " ms" << std::endl;
    writeToLogFile_gpu(oss.str());
    
    printf("Number of selected points: %lu\n", ret_points_log_polar_inner.size());
    
    // Cleanup
    free(save_points);
    free(temp_log_polar_buffer);
    cudaFree(logPolarBuffer);
    cudaFree(d_save_points);
    cudaFree(d_pp_pc);
    cudaFree(d_temp_log_polar_buffer);
    cudaFree(d_temp_log_polar_buffer_filled);
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main(int argc, char* argv[]) {
    printf("=== THEIA ELLIPTICAL LOG-POLAR PROCESSING ===\n");
    printf("Elliptical parameters: a=%.1f (horizontal), b=%.1f (vertical)\n", 
           ELLIPTICAL_A, ELLIPTICAL_B);
    
    // Default parameters
    const char* filename = "longdress_vox10_1052.ply";
    const char* output_filename = "theia_elliptical_output.ply";
    float gaze_pos_x = 0.0f;
    float gaze_pos_y = 1.5f;
    float gaze_pos_z = -1.0f;
    float gaze_dir_x = 0.0f;
    float gaze_dir_y = 0.0f;
    float gaze_dir_z = -1.0f;
    
    // Parse command line arguments
    if (argc > 1) filename = argv[1];
    if (argc > 2) output_filename = argv[2];
    if (argc > 3) gaze_pos_x = atof(argv[3]);
    if (argc > 4) gaze_pos_y = atof(argv[4]);
    if (argc > 5) gaze_pos_z = atof(argv[5]);
    if (argc > 6) gaze_dir_x = atof(argv[6]);
    if (argc > 7) gaze_dir_y = atof(argv[7]);
    if (argc > 8) gaze_dir_z = atof(argv[8]);
    
    // Help message
    if (argc > 1 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
        printf("Usage: %s [input_file] [output_file] [gaze_pos_x] [gaze_pos_y] [gaze_pos_z] [gaze_dir_x] [gaze_dir_y] [gaze_dir_z]\n", argv[0]);
        printf("\nElliptical Log-Polar Transformation:\n");
        printf("  a=%.1f: Horizontal compression factor\n", ELLIPTICAL_A);
        printf("  b=%.1f: Vertical compression factor\n", ELLIPTICAL_B);
        printf("  Ratio a/b=%.1f: More compression in horizontal direction\n", ELLIPTICAL_A/ELLIPTICAL_B);
        return 0;
    }
    
    printf("\nConfiguration:\n");
    printf("  Input file: %s\n", filename);
    printf("  Output file: %s\n", output_filename);
    printf("  Gaze Position: (%.2f, %.2f, %.2f)\n", gaze_pos_x, gaze_pos_y, gaze_pos_z);
    printf("  Gaze Direction: (%.2f, %.2f, %.2f)\n", gaze_dir_x, gaze_dir_y, gaze_dir_z);
    printf("  Elliptical: a/b = %.1f/%.1f = %.1fx horizontal compression\n\n", 
           ELLIPTICAL_A, ELLIPTICAL_B, ELLIPTICAL_A/ELLIPTICAL_B);
    
    // Read point cloud
    const int maxPoints = 1500000;
    Point* points = (Point*)malloc(maxPoints * sizeof(Point));
    int numPoints = readPLY(filename, points, maxPoints);
    
    if (numPoints < 0) {
        printf("Error reading PLY file: %s\n", filename);
        return 1;
    }
    
    printf("Read %d points\n", numPoints);
    
    // Mirror detection and correction
    int count_pos_z = 0, count_neg_z = 0;
    for (int i = 0; i < numPoints; i++) {
        if (points[i].z > 0) count_pos_z++;
        else count_neg_z++;
    }
    
    printf("Z-coordinate analysis: Positive=%d, Negative=%d\n", count_pos_z, count_neg_z);
    
    if (count_neg_z > count_pos_z * 2) {
        printf(">>> Mirroring detected! Inverting Z coordinates...\n");
        for (int i = 0; i < numPoints; i++) {
            points[i].z = -points[i].z;
        }
    }
    
    // Setup gaze
    Gaze gaze;
    gaze.position.x = gaze_pos_x;
    gaze.position.y = gaze_pos_y;
    gaze.position.z = gaze_pos_z;
    gaze.direction.x = gaze_dir_x;
    gaze.direction.y = gaze_dir_y;
    gaze.direction.z = gaze_dir_z;
    
    // Convert to BYTE format
    size_t dataLen = 15 + numPoints * (3 * sizeof(short) + 3 * sizeof(char));
    BYTE* pp_pc = (BYTE*)malloc(dataLen * sizeof(BYTE));
    memset(pp_pc, 0, 15);
    
    // Copy positions
    for (int i = 0; i < numPoints; i++) {
        short x = (short)(points[i].x * 1000.0f);
        short y = (short)(points[i].y * 1000.0f);
        short z = (short)(points[i].z * 1000.0f);
        memcpy(pp_pc + 15 + i * 3 * sizeof(short), &x, sizeof(short));
        memcpy(pp_pc + 15 + i * 3 * sizeof(short) + sizeof(short), &y, sizeof(short));
        memcpy(pp_pc + 15 + i * 3 * sizeof(short) + 2 * sizeof(short), &z, sizeof(short));
    }
    
    // Copy colors
    int colorOffset = 15 + numPoints * 3 * sizeof(short);
    for (int i = 0; i < numPoints; i++) {
        pp_pc[colorOffset + i * 3] = points[i].r;
        pp_pc[colorOffset + i * 3 + 1] = points[i].g;
        pp_pc[colorOffset + i * 3 + 2] = points[i].b;
    }
    
    // Prepare output vectors
    std::vector<Point> ret_points_inner;
    std::vector<Point> ret_points_outer;
    Point* d_points;
    cudaMalloc((void**)&d_points, numPoints * sizeof(Point));
    
    // Processing parameters
    bool dynamic_skip = false;
    float rate_adapt = 1.0f;
    bool augmentation = false;
    
    printf("Calling get_fovea with elliptical transformation...\n");
    
    uint64_t start = NDKGetTime();
    get_fovea(pp_pc, dataLen, gaze, d_points, 
              ret_points_inner, ret_points_outer,
              dynamic_skip, rate_adapt, augmentation);
    uint64_t end = NDKGetTime();
    
    printf("\nRESULTS:\n");
    printf("  Processing time: %llu ms\n", (end - start));
    printf("  Inner points (foveal): %lu\n", ret_points_inner.size());
    printf("  Outer points: %lu\n", ret_points_outer.size());

    // Cleanup host/GPU allocations performed in main
    cudaFree(d_points);
    free(points);
    free(pp_pc);

    return 0;
}