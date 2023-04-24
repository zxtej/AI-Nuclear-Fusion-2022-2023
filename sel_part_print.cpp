#include "include/traj.h"
void sel_part_print(int n_part[3],
                    float pos1x[2][n_partd], float pos1y[2][n_partd], float pos1z[2][n_partd],
                    float pos0x[2][n_partd], float pos0y[2][n_partd], float pos0z[2][n_partd],
                    float posp[2][n_output_part][3],float KE[2][n_output_part],
                    int m[2][n_partd],float dt[2])
{
    for (int p = 0; p < 2; p++)
    {
#pragma omp  parallel for simd
//#pragma omp distribute parallel for simd
        for (int nprt = 0; nprt < n_output_part; nprt++)
        {
            int nprtd = floor(n_part[p] / n_output_part);
            int n = nprt * max(nprtd, 1);
            if (nprtd == 0 && n >= n_part[p]){
                KE[p][nprt] = 0;
                posp[p][nprt][0] = 0;
                posp[p][nprt][1] = 0;
                posp[p][nprt][2] = 0;
                continue;
            }
            float dpos, dpos2 = 0;
            dpos = (pos1x[p][n] - pos0x[p][n]);
            dpos *= dpos;
            dpos2 += dpos;
            dpos = (pos1y[p][n] - pos0y[p][n]);
            dpos *= dpos;
            dpos2 += dpos;
            dpos = (pos1z[p][n] - pos0z[p][n]);
            dpos *= dpos;
            dpos2 += dpos;
            KE[p][nprt] = 0.5 * m[p][n] * (dpos2) / (e_charge_mass * dt[p] * dt[p]);
            // in units of eV
            posp[p][nprt][0] = pos0x[p][n];
            posp[p][nprt][1] = pos0y[p][n];
            posp[p][nprt][2] = pos0z[p][n];
        }
    }
}

void get_Temp_field(float Temp[4][n_space_divz][n_space_divy][n_space_divx],
                    float pos1x[2][n_partd], float pos1y[2][n_partd], float pos1z[2][n_partd],
                    float pos0x[2][n_partd], float pos0y[2][n_partd], float pos0z[2][n_partd],
                    int n_part[3], float posL[3], float dd[3], float dt[2])
{
    fill(reinterpret_cast<float *>(Temp), reinterpret_cast<float *>(Temp) + n_cells * 4, 0);
    for(int p = 0; p < 2; ++p){
        float dtp = dt[p];
        for(int i = 0; i < n_part[p]; ++i){
            float vx = (pos1x[p][i] - pos0x[p][i]) / dtp;
            float vy = (pos1y[p][i] - pos0y[p][i]) / dtp;
            float vz = (pos1z[p][i] - pos0z[p][i]) / dtp;
            float v = sqrtf(vx * vx + vy * vy + vz * vz);
            //https://en.wikipedia.org/wiki/Thermal_velocity#In_three_dimensions
            unsigned int cx = ((pos1x[p][i] - posL[0]) / dd[0] + .5f);
            unsigned int cy = ((pos1y[p][i] - posL[1]) / dd[1] + .5f);
            unsigned int cz = ((pos1z[p][i] - posL[2]) / dd[2] + .5f);
            Temp[p][cz][cy][cx] += v;
            ++Temp[p + 2][cz][cy][cx]; // Electron count is stored in the 2, deut count in 3
        }
        float *p_temp = reinterpret_cast<float *>(Temp[p]);
        for(int i = 0; i < n_cells; ++i){
            float count = p_temp[i + (n_cells << 1)];
            if(count == 0){
                p_temp[i] = 0;
                continue;
            }
            float avg_v  = p_temp[i] / count; // average velocity = total vel / count
            // Now that we have mean velocity, we use https://en.wikipedia.org/wiki/Thermal_velocity#In_three_dimensions
            p_temp[i] = avg_v * avg_v * pi * (mp[p] * e_mass / kb) / 8;
        }
    }
    float *temp_1d = reinterpret_cast<float *>(Temp);
    for(int i = 0; i < n_cells; ++i){
        float ne = temp_1d[i + n_cells * 2], ni = temp_1d[i + n_cells * 3];
        if(ne == ni && ni == 0) continue;
        temp_1d[i + n_cells * 2] = (ne * temp_1d[i] + ni * temp_1d[i + n_cells]) / (ne + ni); // Weighted combined temperature
    }
}