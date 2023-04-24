#include "include/traj.h"
// Interpolate the value at a given point

void get_densityfields(float currentj[2][3][n_space_divz][n_space_divy][n_space_divx],
                       float np[2][n_space_divz][n_space_divy][n_space_divx],
                       float dp[n_space_divz][n_space_divy][n_space_divx],
                       float npt[n_space_divz][n_space_divy][n_space_divx],
                       int nt[2], float KEtot[2], float posL[3], float posH[3], float dd[3],
                       float pos1x[2][n_partd], float pos1y[2][n_partd], float pos1z[2][n_partd],
                       float pos0x[2][n_partd], float pos0y[2][n_partd], float pos0z[2][n_partd],
                       int q[2][n_partd], float dt[2], int n_part[3],
                       float dj[3][n_space_divz][n_space_divy][n_space_divx],
                       float jc[3][n_space_divz][n_space_divy][n_space_divx])
{
    // find number of particle and current density fields

    nt[0] = 0;
    nt[1] = 0;
    KEtot[0] = 0;
    KEtot[1] = 0;
    // set limits beyond which particle is considered as "lost"
    static const float ddi[3] = {1.f / dd[0], 1.f / dd[1], 1.f / dd[2]}; // precalculate reciprocals
    static float dti[2] = {1.f / dt[0], 1.f / dt[1]};
    dti[0] = 1.f / dt[0], dti[1] = 1.f / dt[1]; // dt might change, so recalculate it

    // cell indices for each particle [2][3][n_parte]
    static auto *ii = static_cast<unsigned int(*)[3][n_parte]>(_aligned_malloc(2 * 3 * n_parte * sizeof(unsigned int), alignment));
    // particle velocity array [2][3][n_parte]
    static auto *v = static_cast<float(*)[3][n_parte]>(_aligned_malloc(2 * 3 * n_parte * sizeof(float), alignment));
    // particle offsets array [2][3][n_parte]
    static auto *offset = static_cast<float(*)[3][n_parte]>(_aligned_malloc(2 * 3 * n_parte * sizeof(float), alignment));

    // center of charge field arrays [2-particle type][3 pos][z][y][x]
    static auto *np_center = static_cast<float(*)[n_space_divz][n_space_divy][n_space_divx][3]>(_aligned_malloc(2 * 3 * n_space_divz * n_space_divy * n_space_divx * sizeof(float), alignment));
    // center of current field arrays [2][3-pos][3-current component][z][y][x]
    static auto *jc_center = static_cast<float(*)[3][n_space_divz][n_space_divy][n_space_divx][3]>(_aligned_malloc(2 * 3 * 3 * n_space_divz * n_space_divy * n_space_divx * sizeof(float), alignment));
    // set fields=0 in preparation// Could split into threads.
    fill(reinterpret_cast<float *>(currentj), reinterpret_cast<float *>(currentj) + n_cells * 2 * 3, 0.f);
    fill(reinterpret_cast<float *>(np), reinterpret_cast<float *>(np) + n_cells * 2, 0.f);
    fill(reinterpret_cast<float *>(jc_center), reinterpret_cast<float *>(jc_center) + n_cells * 2 * 3 * 3, 0.f);
    fill(reinterpret_cast<float *>(np_center), reinterpret_cast<float *>(np_center) + n_cells * 3 * 2, 0.f);

    static auto oblist = new unsigned int[2][n_parte]; // list of out of bound particles
    static auto iblist = new unsigned int[2][n_parte];
    int nob[2]; // number of particles out of bounds
    int nib[2]; // number of particles within bounds
//   cout << "get_density_start\n";
// remove out of bounds points and get x,y,z index of each particle
#pragma omp parallel num_threads(2)
    {
        int p = omp_get_thread_num();
#pragma omp parallel for simd num_threads(nthreads)
        for (unsigned int n = 0; n < n_part[p]; ++n) // get cell indices (x,y,z) a particle belongs to
        {
            float x = (pos1x[p][n] - posL[0]) * ddi[0];
            float y = (pos1y[p][n] - posL[1]) * ddi[1];
            float z = (pos1z[p][n] - posL[2]) * ddi[2];
            ii[p][0][n] = (unsigned int)roundf(x);
            ii[p][1][n] = (unsigned int)roundf(y);
            ii[p][2][n] = (unsigned int)roundf(z);
            offset[p][0][n] = x - roundf(x);
            offset[p][1][n] = y - roundf(y);
            offset[p][2][n] = z - roundf(z);
        }
    }
#pragma omp barrier
#pragma omp parallel num_threads(2)
    {
        int p = omp_get_thread_num();
        nob[p] = 0;
        nib[p] = 0;
        for (unsigned int n = 0; n < n_part[p]; ++n)
        {
            if ((ii[p][0][n] > (n_space_divx - 2)) || (ii[p][1][n] > (n_space_divy - 2)) || (ii[p][2][n] > (n_space_divz - 2)) || (ii[p][0][n] < 1) || (ii[p][1][n] < 1) || (ii[p][2][n] < 1))
            {
                oblist[p][nob[p]++] = n;
            }
            else
            {
                iblist[p][nib[p]++] = n;
            }
        }
        n_part[p] -= nob[p];
        for (unsigned int n = 0; n < nob[p]; ++n)
        {
            if(nib[p] <= 0){ // no new particles to bring forward
                q[p][oblist[p][n]] = 0;
                continue;
            }
            int curr_ob = oblist[p][n], last_ib = iblist[p][--nib[p]];
            pos0x[p][curr_ob] = pos0x[p][last_ib];
            pos0y[p][curr_ob] = pos0y[p][last_ib];
            pos0z[p][curr_ob] = pos0z[p][last_ib];
            pos1x[p][curr_ob] = pos1x[p][last_ib];
            pos1y[p][curr_ob] = pos1y[p][last_ib];
            pos1z[p][curr_ob] = pos1z[p][last_ib];
            ii[p][0][curr_ob] = ii[p][0][last_ib];
            ii[p][1][curr_ob] = ii[p][1][last_ib];
            ii[p][2][curr_ob] = ii[p][2][last_ib];
            q[p][curr_ob] = q[p][last_ib];
            q[p][last_ib] = 0;
        }
        // cout << p << ", " << n_part[p] << endl;
//        cout << p << "number of particles out of bounds " << nob[p] << endl;
    }
#pragma omp barrier
    //   cout << "get_density_checked out of bounds\n";

#pragma omp parallel num_threads(2)
    {
        int p = omp_get_thread_num();
#pragma omp parallel for simd num_threads(nthreads)
        for (int n = 0; n < n_part[p]; ++n)
        {
            v[p][0][n] = (pos1x[p][n] - pos0x[p][n]) * dti[p];
            v[p][1][n] = (pos1y[p][n] - pos0y[p][n]) * dti[p];
            v[p][2][n] = (pos1z[p][n] - pos0z[p][n]) * dti[p];
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
            np_center[p][k][j][i][0] += q[p][n] * offset[p][0][n];
            np_center[p][k][j][i][1] += q[p][n] * offset[p][1][n];
            np_center[p][k][j][i][2] += q[p][n] * offset[p][2][n];

            currentj[p][0][k][j][i] += v[p][0][n];
            jc_center[p][0][k][j][i][0] += v[p][0][n] * offset[p][0][n];
            jc_center[p][1][k][j][i][0] += v[p][0][n] * offset[p][1][n];
            jc_center[p][2][k][j][i][0] += v[p][0][n] * offset[p][2][n];
            currentj[p][1][k][j][i] += v[p][1][n];
            jc_center[p][0][k][j][i][1] += v[p][1][n] * offset[p][0][n];
            jc_center[p][1][k][j][i][1] += v[p][1][n] * offset[p][1][n];
            jc_center[p][2][k][j][i][1] += v[p][1][n] * offset[p][2][n];
            currentj[p][2][k][j][i] += v[p][2][n];
            jc_center[p][0][k][j][i][2] += v[p][2][n] * offset[p][0][n];
            jc_center[p][1][k][j][i][2] += v[p][2][n] * offset[p][1][n];
            jc_center[p][2][k][j][i][2] += v[p][2][n] * offset[p][2][n];
        }
    }

#pragma omp barrier
    // for (int p = 0; p < 2; p++)
#pragma omp parallel num_threads(2)
    {
        int p = omp_get_thread_num();
        smoothscalarfield(np[p], np_center[p]);
    }
#pragma omp barrier
    // calculate center of current density field
    for (int p = 0; p < 2; p++)
        for (int c1 = 0; c1 < 3; c1++)
            for (int c = 0; c < 3; c++)
#pragma omp parallel for simd num_threads(nthreads)
                for (unsigned int i = 0; i < n_cells; i++)
                {
                    (reinterpret_cast<float *>(jc_center[p][c1][c]))[i] /= (reinterpret_cast<float *>(currentj[p][c]))[i];
                }
    auto j0 = (float*)(currentj[0]), j1 = (float*)(currentj[1]), dj_1d = (float*)dj, jc_1d = (float*)jc;
#pragma omp parallel for simd num_threads(nthreads)
    for (unsigned int i = 0; i < n_cells * 3; i++)
    {
        float sum = j0[i] + j1[i];
        dj_1d[i] = sum - jc_1d[i];
        jc_1d[i] = sum;
    }
    auto npt_1d = (float*)npt, np0 = (float*)(np[0]), np1 = (float*)(np[1]), dp_1d = (float*)dp;
#pragma omp parallel for simd num_threads(nthreads)
    for (unsigned int i = 0; i < n_cells; i++)
    {
        float sum = np0[i] + np1[i];
        dp_1d[i] = sum - npt_1d[i];
        npt_1d[i] = sum;
    }
#pragma omp barrier
}
