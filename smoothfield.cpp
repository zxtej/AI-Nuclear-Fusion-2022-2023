#include "include/traj.h"
// Interpolate the value at a given point

void smoothscalarfield(float f[n_space_divz][n_space_divy][n_space_divx],
                       float fc[n_space_divz][n_space_divy][n_space_divx][3])
{
    //  cout << "define NFFT plan" << endl;
    nfftf_plan plan; // Define the NFFT plan

    //  cout << "init NFFT plan" << endl;// Memory allocation is completely done by the init routine.
    nfftf_init_3d(&plan, n_space_divx, n_space_divy, n_space_divz, n_cells);

    //    cout << "fill NFFT plan array with values" << endl;
    for (size_t n = 0; n < n_cells; n++)
    {
        plan.f[n][0] = (reinterpret_cast<float *>(f))[n];
        plan.f[n][1] = 0;
    }
    // cout << "fill NFFT plan x with values" << endl;
    for (size_t n = 0; n < n_cells * 3; n++)
    {
        plan.x[n] = (reinterpret_cast<float *>(fc))[n];
    }

    // cout << "nfft precompute: " << p << endl;
    if (plan.flags & PRE_ONE_PSI)
        nfftf_precompute_one_psi(&plan);

   // cout << "nfft check: ";
    const char *check_error_msg = nfftf_check(&plan);
    if (check_error_msg != 0)
    {
        printf("Invalid nfft parameter: %s\n", check_error_msg);
    }

    //        cout << "Execute NFFT transform plan to get fhat" << endl; //  Execute the forward NFFT transform
    nfftf_adjoint(&plan);
    // make regular equispaced x
    /*
    for (int k = 0; k < n_space_divz; k += 1)
        for (int j = 0; j < n_space_divy; j += 1)
            for (int i = 0; i < n_space_divx; i += 1)
            {
                int n = k * n_space_divy * n_space_divz + j * n_space_divz + i;
                plan.x[3 * n] = -0.5 + (float)i / n_space_divx; // p.x[n][0]=x ..,y,z
                plan.x[3 * n + 1] = -0.5 + (float)j / n_space_divy;
                plan.x[3 * n + 2] = -0.5 + (float)k / n_space_divz;
            }
    */
    // cout << "execute the inverse FFTW plan" << endl;  // Execute the inverse FFT
    nfftf_trafo(&plan);
    //        copy back density
    for (unsigned int n = 0; n < n_cells; n++)
        (reinterpret_cast<float *>(f))[n] = plan.f[n][0] / n_cells; // divide by ncell to normalise
    nfftf_finalize(&plan);
}
