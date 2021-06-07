#include "benchmark_utils.h"

double get_wall_seconds(){
  struct timespec ts;
  current_utc_time(&ts);
  double seconds = ts.tv_sec + (double)ts.tv_nsec/1000000000;
  return seconds;
}

void current_utc_time(struct timespec *ts) {

#ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
  clock_serv_t cclock;
  mach_timespec_t mts;
  host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
  clock_get_time(cclock, &mts);
  mach_port_deallocate(mach_task_self(), cclock);
  ts->tv_sec = mts.tv_sec;
  ts->tv_nsec = mts.tv_nsec;
#else
  clock_gettime(CLOCK_REALTIME, ts);
#endif
}

void petsc_triple_ptr_layout(PetscScalar *q, PetscInt N, PetscInt dofs, PetscScalar ****q_3ptr) {
  PetscInt i, j;
  PetscScalar **b;
  PetscMalloc1(N*sizeof(PetscScalar**)+N*N,q_3ptr);
  b = (PetscScalar**)((*q_3ptr) + N);
  for (j = 0; j < N; j++)
    (*q_3ptr)[j] = b + j*N;

  for (j = 0; j < N; j++) 
    for (i = 0; i < N; i++)
      b[j*N+i] = q + j*N*dofs + i*dofs;
};

void free_triple_ptr(PetscScalar ****q_3ptr) {
  void * dummy = (void *)(*q_3ptr);
  PetscFree(dummy);
};