#include "include/traj.h"
#include <math.h>
#include <complex>
#include <fftw3.h>

// Shorthand for cleaner code
const size_t N0 = n_space_divx2, N1 = n_space_divy2, N2 = n_space_divz2,
        N0N1 = N0 * N1, N0N1_2 = N0N1 / 2,
        N2_c = N2 / 2 + 1; // Dimension to store the complex data, as required by fftw (from their docs)
const size_t n_cells4 = N0 * N1 * N2_c; // NOTE: This is not actually n_cells * 4, there is an additional buffer that fftw requires.

void vector_muls(float *A, float B, int n)
{
    // Create a command queue
    cl::CommandQueue queue(context_g, default_device_g);
    // Create memory buffers on the device for each vector
    cl::Buffer buffer_A(context_g, CL_MEM_READ_WRITE, sizeof(float) * n);

    // Copy the lists to memory buffer
    queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(float) * n, A);

    // Create the OpenCL kernel
    cl::Kernel kernel_add = cl::Kernel(program_g, "vector_muls"); // select the kernel program to run

    // Set the arguments of the kernel
    kernel_add.setArg(0, buffer_A); // the 1st argument to the kernel program
    kernel_add.setArg(1, sizeof(float), &B);
    queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(n), cl::NullRange);
    // read result arrays from the device to main memory
    queue.enqueueReadBuffer(buffer_A, CL_TRUE, 0, sizeof(float) * n, A);
    queue.finish(); // wait for the end of the kernel program
}

// Vector multiplication for complex numbers. Note that this is not in-place.
void vector_muls(fftwf_complex *dst, fftwf_complex *A, fftwf_complex *B, int n)
{
    // Create a command queue
    cl::CommandQueue queue(context_g, default_device_g);
    // Create memory buffers on the device for each vector
    cl::Buffer buffer_A(context_g, CL_MEM_WRITE_ONLY, sizeof(fftwf_complex) * n);
    cl::Buffer buffer_B(context_g, CL_MEM_READ_ONLY, sizeof(fftwf_complex) * n);
    cl::Buffer buffer_C(context_g, CL_MEM_READ_ONLY, sizeof(fftwf_complex) * n);

    // Copy the lists C and B to their respective memory buffers
    queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(fftwf_complex) * n, A);
    queue.enqueueWriteBuffer(buffer_C, CL_TRUE, 0, sizeof(fftwf_complex) * n, B);

    // Create the OpenCL kernel
    cl::Kernel kernel_add = cl::Kernel(program_g, "vector_mul_complex"); // select the kernel program to run

    // Set the arguments of the kernel
    kernel_add.setArg(0, buffer_A); // the 1st argument to the kernel program
    kernel_add.setArg(1, buffer_B);
    kernel_add.setArg(2, buffer_C);

    queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(n), cl::NullRange);
    queue.finish(); // wait for the end of the kernel program
    // read result arrays from the device to main memory
    queue.enqueueReadBuffer(buffer_A, CL_TRUE, 0, sizeof(fftwf_complex) * n, dst);
}
inline void convolution_1d_dot_3d_add(fftwf_complex *arr1, fftwf_complex *arr2, fftwf_complex *dst, int N){
    auto a = reinterpret_cast<complex<float>*>(arr1);
    auto b = reinterpret_cast<complex<float>*>(arr2);
    auto c = reinterpret_cast<complex<float>*>(dst);
    for(int i = 0, j = N, k = N + N; i < N; ++i, ++j, ++k){
        c[k] += a[i] * b[k];
        c[j] += a[i] * b[j];
        c[i] += a[i] * b[i];
    }
}
inline void convolution_1d_dot_1d_inplace(fftwf_complex *arr, fftwf_complex *other, int N){
    auto a = reinterpret_cast<complex<float>*>(arr);
    auto b = reinterpret_cast<complex<float>*>(other);
    for(int i = 0; i < N; ++i) a[i] *= b[i];
}
inline void convolution_cross_3d_inplace(fftwf_complex *arr1, fftwf_complex *arr2, int N){
    auto a = reinterpret_cast<complex<float>*>(arr1);
    auto b = reinterpret_cast<complex<float>*>(arr2);
    complex<float> temp1, temp2, temp3;
    for (int i = 0, j = N, k = N + N; i < N; ++i, ++j, ++k)
    {
        temp1 = a[j] * b[k] - a[k] * b[j];
        temp2 = a[k] * b[i] - a[i] * b[k];
        temp3 = a[i] * b[j] - a[j] * b[i];
        a[i] = temp1;
        a[j] = temp2;
        a[k] = temp3;
    }
}
inline void convolution_cross_3d_add(fftwf_complex *arr1, fftwf_complex *arr2, fftwf_complex *dst, int N){
    auto a = reinterpret_cast<complex<float>*>(arr1);
    auto b = reinterpret_cast<complex<float>*>(arr2);
    auto c = reinterpret_cast<complex<float>*>(dst);
    for (int i = 0, j = N, k = N + N; i < N; ++i, ++j, ++k)
    {
        c[i] += a[j] * b[k] - a[k] * b[j];
        c[j] += a[k] * b[i] - a[i] * b[k];
        c[k] += a[i] * b[j] - a[j] * b[i];
    }
}
inline void convolution_cross_3d(fftwf_complex *arr1, fftwf_complex *arr2, fftwf_complex *dst, int N){
    auto a = reinterpret_cast<complex<float>*>(arr1);
    auto b = reinterpret_cast<complex<float>*>(arr2);
    auto c = reinterpret_cast<complex<float>*>(dst);
    for (int i = 0, j = N, k = N + N; i < N; ++i, ++j, ++k)
    {
        c[i] = a[j] * b[k] - b[k] * a[j];
        c[j] = a[k] * b[i] - b[i] * a[k];
        c[k] = a[i] * b[j] - b[j] * a[i];
    }
}
inline void copy_to_dim2(float *dst, float src[n_space_divz][n_space_divy][n_space_divx]){
    float *ptr = dst;
    for(int k = 0; k < n_space_divz; ++k, ptr += N0N1_2){
        for(int j = 0; j < n_space_divy; ++j, ptr += N0)
            copy(&src[k][j][0], &src[k][j + 1][0], ptr);
    }
}
inline void copy_to_dim2_3d(float *dst, float src[3][n_space_divz][n_space_divy][n_space_divx]){
    for(int c = 0; c < 3; ++c){
        float *ptr = dst + (c * n_cells8);
        for(int k = 0; k < n_space_divz; ++k, ptr += N0N1_2)
            for(int j = 0; j < n_space_divy; ++j, ptr += N0)
                copy(&src[c][k][j][0], &src[c][k][j + 1][0], ptr);
    }
}
inline void copy_to_dim1(float dst[n_space_divz][n_space_divy][n_space_divx], float *src){
    float *ptr = src;
    for(int k = 0; k < n_space_divz; ++k, ptr += N0N1_2)
        for(int j = 0; j < n_space_divy; ++j, ptr += N0)
            move(ptr, ptr + n_space_divx, &dst[k][j][0]);
}
inline void copy_to_dim1_3d(float dst[3][n_space_divz][n_space_divy][n_space_divx], float *src){
    for(int c = 0; c < 3; ++c){
        float *ptr = src + (c * n_cells8);
        for(int k = 0; k < n_space_divz; ++k, ptr += N0N1_2)
            for(int j = 0; j < n_space_divy; ++j, ptr += N0)
                move(ptr, ptr + n_space_divx, &dst[c][k][j][0]);
    }
}
inline void fma_to_dim1_3d(float dst[3][n_space_divz][n_space_divy][n_space_divx], float *src, float scl){
    const __m256 scl_V = _mm256_set1_ps(scl);
    for(int c = 0; c < 3; ++c){
        float *ptr = src + (c * n_cells8);
        for(int k = 0; k < n_space_divz; ++k, ptr += N0N1_2)
            for(int j = 0; j < n_space_divy; ++j, ptr += N0)
                for(int i = 0; i < n_space_divx; i += 8) // Experimental AVX2 code
                     _mm256_store_ps(dst[c][k][j], _mm256_fmadd_ps(_mm256_load_ps(ptr), scl_V, _mm256_load_ps(dst[c][k][j])));
                //for(int i = 0; i < n_space_divx; ++i)
                //    dst[c][k][j][i] = fmaf(*(ptr + i), scl, dst[c][k][j][i]);
    }
}
bool checkInRange(string name, float data[3][n_space_divz][n_space_divy][n_space_divx], float minval, float maxval){
    bool toolow = false, toohigh = false;
    const float *data_1d = reinterpret_cast<float *>(data);
    for (unsigned int i = 0; i < n_cells * 3; ++i)
    {
        toolow |= data_1d[i] < minval;
        toohigh |= data_1d[i] > maxval;
    }
    if (toolow)
    {
        const float *minelement = min_element(data_1d, data_1d + 3 * n_cells);
        size_t pos = minelement - &data[0][0][0][0];
        int count = 0;
        for (unsigned int n = 0; n < n_cells * 3; ++n)
            count += data_1d[n] < minval;
        int x, y, z;
        id_to_cell(pos, &x, &y, &z);
        char mode = z < n_space_divz ? 'x' : z < n_space_divz2 ? 'y' : 'z';
        z %= n_space_divz;
        cout << "Min " << name << mode << ": " << *minelement<< " (" << x << "," << y << "," << z << ") (" << count << " values below threshold)\n";
    }
    if (toohigh)
    {
        const float *maxelement = max_element(data_1d, data_1d + 3 * n_cells);
        size_t pos = maxelement - &data[0][0][0][0];
        int count = 0;
        for (unsigned int n = 0; n < n_cells * 3; ++n)
            count += data_1d[n] > maxval;
        int x, y, z;
        id_to_cell(pos, &x, &y, &z);
        char mode = z < n_space_divz ? 'x' : z < n_space_divz2 ? 'y' : 'z';
        z %= n_space_divz;
        cout << "Max " << name << mode << ": " << *maxelement<< " (" << x << "," << y << "," << z << ") (" << count << " values above threshold)\n";
    }
    return toolow || toohigh;
}

// Arrays for fft, output is multiplied by 2 because the convolution pattern should be double the size
// (a particle interacts both 64 cells up and 64 cells down, so we need 128 cells to calculate this information)
auto *fft_real = reinterpret_cast<float(&)[3][n_cells8]>(*fftwf_alloc_real(n_cells8 * 3));
auto *fft_complex = reinterpret_cast<fftwf_complex(&)[6][n_cells4]>(*fftwf_alloc_complex(n_cells4 * 6));

// pre-calculate 1/r2, 1/r3 etc to make it faster to calculate electric and magnetic fields
// they are the r/r things found in https://en.wikipedia.org/wiki/Jefimenko%27s_equations#Equations
// in particular, [0] -> 1/|r|, [1] -> rx/|r|^2, [2] -> rx/|r|^3, where x is any of the 3 components (x, y, z)
// these are also scaled by: [0] -> e_charge * rpartspart * -1/c^2, [1] -> e_charge * rpartspart * 1/c, [2] -> e_charge * rpartspart
auto *r_kernels = reinterpret_cast<fftwf_complex(&)[3][3][n_cells4]>(*fftwf_alloc_complex(3 * 3 * n_cells4));

fftwf_plan
    planForward_p_dp,
    planForward_j,
    planForward_dj,
    planBackward_E,
    planBackward_B,
    planBackward_V;

void get_EBV_plans(float dd[3]){
    // allocate and initialize to 0
    auto kernel_base = new float[3][3][N2][N1][N0]; 
    int dims[3] = {N0, N1, N2};

    // Create fftw plans    
    fftwf_plan planfor_k = fftwf_plan_many_dft_r2c(3, dims, 9, reinterpret_cast<float *>(kernel_base[0][0]), NULL, 1, n_cells8, reinterpret_cast<fftwf_complex *>(r_kernels[0][0]), NULL, 1, n_cells4, FFTW_ESTIMATE);
    planForward_j = fftwf_plan_many_dft_r2c(3, dims, 3, fft_real[0], NULL, 1, n_cells8, fft_complex[0], NULL, 1, n_cells4, FFTW_MEASURE);
    planForward_dj = fftwf_plan_many_dft_r2c(3, dims, 3, fft_real[0], NULL, 1, n_cells8, fft_complex[3], NULL, 1, n_cells4, FFTW_MEASURE);
    planForward_p_dp = fftwf_plan_many_dft_r2c(3, dims, 2, fft_real[0], NULL, 1, n_cells8, fft_complex[0], NULL, 1, n_cells4, FFTW_MEASURE);
    planBackward_B = fftwf_plan_many_dft_c2r(3, dims, 3, fft_complex[0], NULL, 1, n_cells4, fft_real[0], NULL, 1, n_cells8, FFTW_MEASURE);
    planBackward_E = fftwf_plan_many_dft_c2r(3, dims, 3, fft_complex[3], NULL, 1, n_cells4, fft_real[0], NULL, 1, n_cells8, FFTW_MEASURE);
    planBackward_V = fftwf_plan_dft_c2r_3d(dims[0], dims[1], dims[2], fft_complex[0], fft_real[0], FFTW_MEASURE);
    cout << "planning done\n";
    float r1, r2, r3, rx, ry, rz, rx2, ry2, rz2;
    int i, j, k, loc_i, loc_j, loc_k;
    // precalculate 1/r^3 (field) and 1/r^2 (energy)
    for (k = -n_space_divz; k < n_space_divz; k++)
    {
        loc_k = k + (k < 0 ? n_space_divz2 : 0); // The "logical" array position
        // We wrap around values smaller than 0 to the other side of the array, since 0, 0, 0 is defined as the center of the convolution pattern an hence rz should be 0
        rz = k * dd[2]; // The change in z coordinate for the k-th cell.
        rz2 = rz * rz;
        for (j = -n_space_divy; j < n_space_divy; j++)
        {
            loc_j = j + (j < 0 ? n_space_divy2 : 0);
            ry = j * dd[1];
            ry2 = ry * ry + rz2;
            for (i = -n_space_divx; i < n_space_divx; i++)
            {
                loc_i = i + (i < 0 ? n_space_divx2 : 0);
                rx = i * dd[0];
                rx2 = rx * rx + ry2;
                r1 = rx2 == 0? 0.f : powf(rx2, -0.5);
                kernel_base[0][0][loc_k][loc_j][loc_i] = r1;
                kernel_base[0][1][loc_k][loc_j][loc_i] = r1;
                kernel_base[0][2][loc_k][loc_j][loc_i] = r1;
                r2 = rx2 == 0? 0.f : powf(rx2, -1.0);
                kernel_base[1][0][loc_k][loc_j][loc_i] = r2 * rx;
                kernel_base[1][1][loc_k][loc_j][loc_i] = r2 * ry;
                kernel_base[1][2][loc_k][loc_j][loc_i] = r2 * rz;
                r3 = rx2 == 0? 0.f : powf(rx2, -1.5);
                kernel_base[2][0][loc_k][loc_j][loc_i] = r3 * rx;
                kernel_base[2][1][loc_k][loc_j][loc_i] = r3 * ry;
                kernel_base[2][2][loc_k][loc_j][loc_i] = r3 * rz;
            }
        }
    }
    // Multiply by the respective constants here, since it is faster to parallelize it
    vector_muls(reinterpret_cast<float *>(kernel_base[0]), -e_charge * r_part_spart / n_cells8 / c_light / c_light, n_cells8 * 3);
    vector_muls(reinterpret_cast<float *>(kernel_base[1]),  e_charge * r_part_spart / n_cells8 / c_light          , n_cells8 * 3);
    vector_muls(reinterpret_cast<float *>(kernel_base[2]),  e_charge * r_part_spart / n_cells8                    , n_cells8 * 3);
    cout << "kernel real done\n";

    fftwf_execute(planfor_k); // r_kernel = fft(kernel_base)
    fftwf_destroy_plan(planfor_k);
    delete[] kernel_base;
    cout << "kernel fft done\n";
}

int calcEBV(float V[n_space_divz][n_space_divy][n_space_divx],
           float E[3][n_space_divz][n_space_divy][n_space_divx], float B[3][n_space_divz][n_space_divy][n_space_divx],
           float Ee[3][n_space_divz][n_space_divy][n_space_divx], float Be[3][n_space_divz][n_space_divy][n_space_divx],
           float dp[n_space_divz][n_space_divy][n_space_divx], float chargep[n_space_divz][n_space_divy][n_space_divx],
           float dj[3][n_space_divz][n_space_divy][n_space_divx], float jc[3][n_space_divz][n_space_divy][n_space_divx], float dd[3],
           float Emax, float Bmax, float dt)
{
    // Scale current and charge by dt first
    {
        float dt_1 = 0; //1.f / dt;
        auto dp1 = reinterpret_cast<float*>(dp), dj1 = reinterpret_cast<float*>(dj);
        for(int i = 0; i < n_cells; ++i) dp1[i] *= dt_1;
        for(int i = 0; i < n_cells * 3; ++i) dj1[i] *= dt_1;
    }
    // Uses Jefimenko's equations, see https://en.wikipedia.org/wiki/Jefimenko%27s_equations#Equations
    // Note that we cannot do the curl things for Ee and Be
    // Push J into complex[0-2] and dJ into complex[3-5]
    fill(fft_real[0], fft_real[3], 0.f);
    copy_to_dim2_3d(fft_real[0], jc);
    fftwf_execute(planForward_j);
    fill(fft_real[0], fft_real[3], 0.f);
    copy_to_dim2_3d(fft_real[0], dj);
    fftwf_execute(planForward_dj);

    // Let complex[0-2] be the place where we add everything up to prepare for IFFT to B
    convolution_cross_3d_inplace(fft_complex[0], r_kernels[2][0], n_cells4); // J cross r/r^3
    convolution_cross_3d_add(fft_complex[3], r_kernels[1][0], fft_complex[0], n_cells4); // dJ cross r/r^2
    fftwf_execute(planBackward_B);
    copy_to_dim1_3d(B, fft_real[0]);

    // Let complex[3-5] be the place where we add everything up to prepare for IFFT to E
    // dJ dot 1/r
    convolution_1d_dot_1d_inplace(fft_complex[3], r_kernels[0][0], n_cells4);
    convolution_1d_dot_1d_inplace(fft_complex[4], r_kernels[0][0], n_cells4);
    convolution_1d_dot_1d_inplace(fft_complex[5], r_kernels[0][0], n_cells4);

    fill(fft_real[0], fft_real[2], 0.f);
    // Push p to complex[0], dp to complex[1]
    copy_to_dim2(fft_real[0], chargep);
    copy_to_dim2(fft_real[1], dp);
    fftwf_execute(planForward_p_dp);

    // Convolution, adding to complex[3-5]
    convolution_1d_dot_3d_add(fft_complex[0], r_kernels[2][0], fft_complex[3], n_cells4); // Add p dot r/r^3
    convolution_1d_dot_3d_add(fft_complex[1], r_kernels[1][0], fft_complex[3], n_cells4); // Add dp dot r/r^2
    fftwf_execute(planBackward_E);
    copy_to_dim1_3d(E, fft_real[0]);

    // Now get V
    convolution_1d_dot_1d_inplace(fft_complex[0], r_kernels[0][0], n_cells4);
    fftwf_execute(planBackward_V);
    #ifdef Uon_
    {
        // Scale V correctly - it is currently off by -1/c^2. We need it to be k.
        constexpr float scl_factor = -c_light * c_light * kc;
        uint64_t loc = 0;
        for(int k = 0; k < n_space_divz; ++k, loc += N0N1_2)
            for(int j = 0; j < n_space_divy; ++j, loc += N0)
                for(int i = 0; i < n_space_divx; ++i)
                    V[k][j][i] = fft_real[0][loc + i] * scl_factor;
    }
    #endif

    // Scale vectors by the appropriate amount
    // And add on Ee and Be
    auto E_1d = reinterpret_cast<float*>(E);
    auto B_1d = reinterpret_cast<float*>(B);
    auto Ee_1d = reinterpret_cast<float*>(Ee);
    auto Be_1d = reinterpret_cast<float*>(Be);
    {
        const float scl_factor = u0 / 4 / pi;
        for(int i = 0; i < n_cells * 3; ++i){
            E_1d[i] = fmaf(E_1d[i], kc, Ee_1d[i]);
            B_1d[i] = fmaf(B_1d[i], scl_factor, Be_1d[i]);
        }
    }
    // Check if fields are out of safe range. Too high fields will result in large forces that could cause instability
    // Hence if fields are too large, decrease the timestep
    bool E_exceeds = checkInRange("E", E, -Emax, Emax), B_exceeds = checkInRange("B", B, -Bmax, Bmax);
    return E_exceeds || B_exceeds;
}