#include "include/traj.h"

void id_to_cell(int id, int*x, int*y, int*z){
    *x = id % n_space_divx;
    id = id / n_space_divx;
    *y = id % n_space_divy;
    *z = id / n_space_divy;
}

void Time::mark(){
    marks.push_back(chrono::high_resolution_clock::now());
}

float Time::elapsed(){
    unsigned long long time = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - marks.back()).count();
    marks.pop_back();
    return (float)time * 1e-6;
}

// Get the same result as elapsed, but also insert the current time point back in
float Time::replace(){
    auto now = chrono::high_resolution_clock::now();
    auto back = marks.back();
    unsigned long long time = chrono::duration_cast<chrono::microseconds>(now - back).count();
    back = now;
    return (float)time * 1e-6;
}

Log::Log() { if(!log_file.is_open()) log_file.open("log.csv"); log_file << setprecision(5); }

void Log::newline(){
    log_file << "\n";
    log_file.flush();
    firstEntry = true;
}
void Log::close(){
    log_file.close();
}


void log_headers()
{
    logger.write("t_large");
    logger.write("t_small");
    logger.write("dt_ch");
    logger.write("nc_ele");
    logger.write("nc_deut");
    logger.write("dt_ele");
    logger.write("dt_deut");
    logger.write("t_sim");
    logger.write("ne");
    logger.write("ni");
    logger.write("KEt_e");
    logger.write("KEt_d");
    logger.write("Ele_pot");
    logger.write("Mag_pot");
    logger.write("E_tot");
    logger.newline();
}


void log_entry(int i_time, int ntime, int cdt, int total_ncalc[2], float dt[2], double t, int nt[2], float KEtot[2], float U[2])
{
    logger.write(i_time);
    logger.write(ntime);
    logger.write(cdt);
    logger.write(total_ncalc[0]);
    logger.write(total_ncalc[1]);
    logger.write(dt[0]);
    logger.write(dt[1]);
    logger.write(t);
    logger.write(nt[0]);
    logger.write(nt[1]);
    logger.write(KEtot[0]);
    logger.write(KEtot[1]);
    logger.write(U[0]);
    logger.write(U[1]);
    logger.write(KEtot[0] + KEtot[1] + U[0] + U[1]);
    logger.newline();
}