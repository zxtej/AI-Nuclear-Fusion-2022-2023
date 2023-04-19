#include "include/traj.h"
// Interpolate the value at a given point

void get_densityfields(float currentj[2][3][n_space_divz][n_space_divy][n_space_divx],
                       float np[2][n_space_divz][n_space_divy][n_space_divx],
                       float npt[n_space_divz][n_space_divy][n_space_divx],
                       int nt[2], float KEtot[2], float posL[3], float posH[3], float dd[3],
                       float pos1x[2][n_partd], float pos1y[2][n_partd], float pos1z[2][n_partd],
                       float pos0x[2][n_partd], float pos0y[2][n_partd], float pos0z[2][n_partd],
                       int q[2][n_partd], float dt[2], int n_part[3],
                       float jc[3][n_space_divz][n_space_divy][n_space_divx])
{
    // find number of particle and current density fields

    nt[0] = 0;
    nt[1] = 0;
    KEtot[0] = 0;
    KEtot[1] = 0;
    // set limits beyond which particle is considered as "lost"
    static const float ddi[3] = {1.f / dd[0], 1.f / dd[1], 1.f / dd[2]}; // precalculate reciprocals
    static const float dti[2] = {1.f / dt[0], 1.f / dt[1]};

    // cell indices for each particle [2][3][n_parte]
    auto *ii = static_cast<unsigned int(*)[3][n_parte]>(_aligned_malloc(2 * 3 * n_parte * sizeof(unsigned int), alignment));
    // particle velocity array [2][3][n_parte]
    static auto *v = static_cast<float(*)[3][n_parte]>(_aligned_malloc(2 * 3 * n_parte * sizeof(float), alignment));
    // particle offsets array [2][3][n_parte]
    static auto *offset = static_cast<float(*)[3][n_parte]>(_aligned_malloc(2 * 3 * n_parte * sizeof(float), alignment));

    // center of charge field arrays [2-particle type][3 pos][z][y][x]
    auto *np_center = static_cast<float(*)[3][n_space_divz][n_space_divy][n_space_divx]>(_aligned_malloc(2 * 3 * n_space_divz * n_space_divy * n_space_divx * sizeof(float), alignment));
    // center of current field arrays [2][3-pos][3-current component][z][y][x]
    auto *jc_center = static_cast<float(*)[3][3][n_space_divz][n_space_divy][n_space_divx]>(_aligned_malloc(2 * 3 * 3 * n_space_divz * n_space_divy * n_space_divx * sizeof(float), alignment));
    // set fields=0 in preparation// Could split into threads.
    fill(reinterpret_cast<float *>(currentj), reinterpret_cast<float *>(currentj) + n_cells * 2 * 3, 0.f);
    fill(reinterpret_cast<float *>(np), reinterpret_cast<float *>(np) + n_cells * 2, 0.f);
    fill(reinterpret_cast<float *>(jc_center), reinterpret_cast<float *>(jc_center) + n_cells * 2 * 3 * 3, 0.f);
    fill(reinterpret_cast<float *>(np_center), reinterpret_cast<float *>(np_center) + n_cells * 3 * 2, 0.f);

    auto oblist = new unsigned int[2][n_parte]; // list of out of bound particles
    auto iblist = new unsigned int[2][n_parte];
    int nob[2]; // number of particles out of bounds
    int nib[2]; // number of particles within bounds
//   cout << "get_density_start\n";
// remove out of bounds points and get x,y,z index of each particle
#pragma omp parallel num_threads(2)
    {
        int p = omp_get_thread_num();
        // get cell indices (x,y,z) a particle belongs to
#pragma omp parallel for simd num_threads(nthreads)
        for (unsigned int n = 0; n < n_part[p]; ++n)
        {
            ii[p][0][n] = (int)((pos1x[p][n] - posL[0]) * ddi[0]);
            ii[p][1][n] = (int)((pos1y[p][n] - posL[1]) * ddi[1]);
            ii[p][2][n] = (int)((pos1z[p][n] - posL[2]) * ddi[2]);
        }
#pragma omp barrier
        nob[p] = 0;
        nib[p] = 0;
        for (unsigned int n = 0; n < n_part[p]; ++n)
        {
            if ((ii[p][0][n] > (n_space_divx - 2)) || (ii[p][1][n] > (n_space_divy - 2)) || (ii[p][2][n] > (n_space_divz - 2)) || (ii[p][0][n] < 1) || (ii[p][1][n] < 1) || (ii[p][2][n] < 1))
            {
                oblist[p][nob[p]] = n;
                nob[p]++;
            }
            else
            {
                iblist[p][nob[p]] = n;
                nib[p]++;
            }
        }
        for (unsigned int n = 0; n < nob[p]; ++n)
        {
            n_part[p]--;
            nib[p]--;
            int last = iblist[p][nib[p]];
            pos0x[p][oblist[p][n]] = pos0x[p][last];
            pos0y[p][oblist[p][n]] = pos0y[p][last];
            pos0z[p][oblist[p][n]] = pos0z[p][last];
            pos1x[p][oblist[p][n]] = pos1x[p][last];
            pos1y[p][oblist[p][n]] = pos1y[p][last];
            pos1z[p][oblist[p][n]] = pos1z[p][last];
            ii[p][0][oblist[p][n]] = ii[p][0][last];
            ii[p][1][oblist[p][n]] = ii[p][1][last];
            ii[p][2][oblist[p][n]] = ii[p][2][last];
            q[p][n] = q[p][last];
            q[p][last] = 0;
        }

        //      cout << p << ", " << n_part[p] << endl;
        //        cout <<p <<"number of particles out of bounds " << nob[p]<<endl;
    }
#pragma omp barrier
    // cout << "get_density_checked out of bounds\n";

#pragma omp parallel num_threads(2)
    {
        int p = omp_get_thread_num();
#pragma omp parallel for simd num_threads(nthreads)
        for (int n = 0; n < n_part[p]; ++n)
        {
            v[p][0][n] = (pos1x[p][n] - pos0x[p][n]) * dti[p];
            v[p][1][n] = (pos1y[p][n] - pos0y[p][n]) * dti[p];
            v[p][2][n] = (pos1z[p][n] - pos0z[p][n]) * dti[p];
            offset[p][0][n] = pos1x[p][n] * ddi[0] - (float)((int)ii[p][0][n] - n_space_divx / 2 + 1);
            offset[p][1][n] = pos1y[p][n] * ddi[1] - (float)((int)ii[p][1][n] - n_space_divy / 2 + 1);
            offset[p][2][n] = pos1z[p][n] * ddi[2] - (float)((int)ii[p][2][n] - n_space_divz / 2 + 1);
            /*
            if (p == 0)
            {
                if (offset[p][0][n] < -0.5f)
                    cout << offset[p][0][n] << ", " << pos1x[p][n] << ", " << ddi[0] << ", " << (float)ii[p][0][n] << endl;
                if (offset[p][0][n] > -0.5f)
                    cout << offset[p][0][n] << ", " << pos1x[p][n] << ", " << ddi[0] << ", " << (float)ii[p][0][n] << endl;
            }
            */
        }

        // #pragma omp parallel for simd num_threads(nthreads) reduction (+: KEtot[0] ,nt[0],KEtot[1] ,nt[1] )
        for (int n = 0; n < n_part[p]; ++n)
        {
            KEtot[p] += v[p][0][n] * v[p][0][n] + v[p][1][n] * v[p][1][n] + v[p][2][n] * v[p][2][n];
            nt[p] += q[p][n];
        }
        KEtot[p] *= 0.5 * mp[p] / (e_charge_mass)*r_part_spart; // as if these particles were actually samples of the greater thing
#pragma omp barrier
#pragma omp parallel for simd num_threads(nthreads)
        for (int n = 0; n < n_part[p]; ++n)
        {
            v[p][0][n] *= q[p][n];
            v[p][1][n] *= q[p][n];
            v[p][2][n] *= q[p][n];
        }

        for (int n = 0; n < n_part[p]; ++n)
        {
            unsigned int i = ii[p][0][n], j = ii[p][1][n], k = ii[p][2][n];
            np[p][k][j][i] += q[p][n];
            np_center[p][0][k][j][i] += q[p][n] * offset[p][0][n];
            np_center[p][1][k][j][i] += q[p][n] * offset[p][1][n];
            np_center[p][2][k][j][i] += q[p][n] * offset[p][2][n];

            currentj[p][0][k][j][i] += v[p][0][n];
            jc_center[p][0][0][k][j][i] += v[p][0][n] * offset[p][0][n];
            jc_center[p][1][0][k][j][i] += v[p][0][n] * offset[p][1][n];
            jc_center[p][2][0][k][j][i] += v[p][0][n] * offset[p][2][n];
            currentj[p][1][k][j][i] += v[p][1][n];
            jc_center[p][0][1][k][j][i] += v[p][1][n] * offset[p][0][n];
            jc_center[p][1][1][k][j][i] += v[p][1][n] * offset[p][1][n];
            jc_center[p][2][1][k][j][i] += v[p][1][n] * offset[p][2][n];
            currentj[p][2][k][j][i] += v[p][2][n];
            jc_center[p][0][2][k][j][i] += v[p][2][n] * offset[p][0][n];
            jc_center[p][1][2][k][j][i] += v[p][2][n] * offset[p][1][n];
            jc_center[p][2][2][k][j][i] += v[p][2][n] * offset[p][2][n];
        }
    }

#pragma omp barrier
    cout << "get_density_almost done\n";
    // calculate center of charge field
    for (int c = 0; c < 3; c++)
        cout << ddi[c] << ", ";
    cout << endl;

    cout << "calculate center of charge field" << endl;
    for (int p = 0; p < 2; p++)
        for (int c = 0; c < 3; c++)
            // #pragma omp parallel for simd num_threads(nthreads)
            for (unsigned int i = 0; i < n_cells; i++)
            {
                (reinterpret_cast<float *>(np_center[p][c]))[i] = (reinterpret_cast<float *>(np_center[p][c]))[i] / (((reinterpret_cast<float *>(np[p]))[i]) + 1.0e-10f);
            }
    int p = 0;
    for (int i = 0; i < n_space_divx; i += 1)
    {
        for (int j = 0; j < n_space_divy; j += 1)
        {
            for (int k = 0; k < n_space_divz; k += 1)
            {

                // int n = i * n_space_divy * n_space_divz + j * n_space_divz + k;
                np_center[p][0][k][j][i] = (np_center[p][0][k][j][i] / (np[p][k][j][i] + 1e-20f) + i) / n_space_divx;
                np_center[p][1][k][j][i] = (np_center[p][0][k][j][i] / (np[p][k][j][i] + 1e-20f) + j) / n_space_divy;
                np_center[p][2][k][j][i] = (np_center[p][0][k][j][i] / (np[p][k][j][i] + 1e-20f) + k) / n_space_divz;
                //               (reinterpret_cast<float *>(np_center[p][c]))[n] = (reinterpret_cast<float *>(np_center[p][c]))[n] / (((reinterpret_cast<float *>(np[p]))[n]) + 1.0e-10f);
            }
        }
    }
/*
    // Print out the center of charge  grid values
    for (int i = 0; i < n_space_divx; i += 8)
    {
        for (int j = 0; j < n_space_divy; j += 8)
        {
            for (int k = 0; k < n_space_divz; k += 8)
            {
                std::cout << (reinterpret_cast<float *>(np_center[0][1]))[i * n_space_divy * n_space_divz + j * n_space_divz + k] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
*/
#pragma omp barrier

    // Allocate memory for the output grid
    cout << "allocate memory for output grid" << endl;
    auto *output = new fftwf_complex[n_cells];

    // Define the NFFT plan
    cout << "define NFFT plan" << endl;
    nfftf_plan plan;

    // Memory allocation is completely done by the init routine.
    cout << "init NFFT plan" << endl;
    nfftf_init_3d(&plan, n_space_divx, n_space_divy, n_space_divz, n_cells);

    cout << "fill NFFT plan array with values" << endl;
    for (unsigned int i = 0; i < n_cells; i++)
    {
        plan.f[i][0] = (reinterpret_cast<float *>(np[0]))[i];
        plan.f[i][1] = 0;
    }

    cout << "fill NFFT plan x with values" << endl;
    for (int i = 0; i < n_space_divx; i += 1)
    {
        for (int j = 0; j < n_space_divy; j += 1)
        {
            for (int k = 0; k < n_space_divz; k += 1)
            {
                int n = k * n_space_divy * n_space_divz + j * n_space_divz + i;
                for (int c = 0; c < 3; c++)
                    plan.x[n * 3 + c] = np_center[0][0][k][j][i];
                //                    plan.x[n * 3 + c] = (reinterpret_cast<float *>(np_center[0][c]))[n];
            }
        }
    }

    /*
// Print out the non-equispaced grid values
for (int i = 0; i < n_space_divx; i += 4)
{
    for (int j = 0; j < n_space_divy; j += 4)
    {
        for (int k = 0; k < n_space_divz; k += 4)
        {
            std::cout << plan.f[i * n_space_divy * n_space_divz + j * n_space_divz + k][0] << " ";
            // std::cout << plan.x[(i * n_space_divy * n_space_divz + j * n_space_divz + k)*3] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
*/
    //  Execute the forward NFFT transform
    cout << " NFFT transform forward plan" << endl;
    nfftf_adjoint_3d(&plan);
    // nfftf_trafo_3d(&plan);
    cout << "copy the forward output " << endl;
    for (int i = 0; i < n_space_divx; i += 1)
    {
        for (int j = 0; j < n_space_divy; j += 1)
        {
            for (int k = 0; k < n_space_divz; k += 1)
            {
                int n = i * n_space_divy * n_space_divz + j * n_space_divz + k;
                if (plan.f[n][0])
                    std::cout << "f " << i << "," << j << "," << k << "," << plan.f[n][0] << " ";
                if (plan.f_hat[n][0])
                    std::cout << "f_hat " << i << "," << j << "," << k << "," << plan.f_hat[n][0] << " ";
                // std::cout << plan.x[(i * n_space_divy * n_space_divz + j * n_space_divz + k)*3] << " ";
            }
            //   std::cout << std::endl;
        }
        //  std::cout << std::endl;
    }
    for (unsigned int i = 0; i < n_cells; i++)
    {
        output[i][0] = plan.f_hat[i][0];
        output[i][1] = plan.f_hat[i][1];
    }
    // Define the FFTW plan for inverse FFT
    cout << "define FFTW plan" << endl;

    fftwf_plan ifft = fftwf_plan_dft_3d(n_space_divx, n_space_divy, n_space_divz, reinterpret_cast<fftwf_complex *>(output),
                                        reinterpret_cast<fftwf_complex *>(output),
                                        FFTW_BACKWARD, FFTW_ESTIMATE);

    // Execute the inverse FFT
    cout << "execute the inverse FFTW plan" << endl;

    fftwf_execute(ifft);
    /*
        // Print out the equispaced grid values
        for (int i = 0; i < n_space_divx; i += 4)
        {
            for (int j = 0; j < n_space_divy; j += 4)
            {
                for (int k = 0; k < n_space_divz; k += 4)
                {
                    std::cout << output[i * n_space_divy * n_space_divz + j * n_space_divz + k][0] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        */
    nfftf_finalize(&plan);
    fftwf_destroy_plan(ifft);

    // calculate center of current density field
    for (int p = 0; p < 2; p++)
        for (int c1 = 0; c1 < 3; c1++)
            for (int c = 0; c < 3; c++)
#pragma omp parallel for simd num_threads(nthreads)
                for (unsigned int i = 0; i < n_cells; i++)
                {
                    (reinterpret_cast<float *>(jc_center[p][c1][c]))[i] /= (reinterpret_cast<float *>(currentj[p][c]))[i];
                }
#pragma omp parallel for simd num_threads(nthreads)
    for (unsigned int i = 0; i < n_cells * 3; i++)
    {
        (reinterpret_cast<float *>(jc))[i] = (reinterpret_cast<float *>(currentj[0]))[i] + (reinterpret_cast<float *>(currentj[1]))[i];
    }
#pragma omp parallel for simd num_threads(nthreads)
    for (unsigned int i = 0; i < n_cells; i++)
    {
        (reinterpret_cast<float *>(npt))[i] = (reinterpret_cast<float *>(np[0]))[i] + (reinterpret_cast<float *>(np[1]))[i];
    }
#pragma omp barrier
}
