#define PROBLEM_TYPE_1D_O2

#include<petsc.h>
#include "appctx.h"

PetscErrorCode setup_timestepper(TimeCtx& timectx, PetscScalar tau) 
{
  int i,j;
  
  timectx.HI_el[0] = tau*4.389152966531085*2/timectx.Tpb;
  timectx.HI_el[1] = tau*-1.247624770988935*2/timectx.Tpb;
  timectx.HI_el[2] = tau*0.614528095966794*2/timectx.Tpb;
  timectx.HI_el[3] = tau*-0.327484862937516*2/timectx.Tpb;

  timectx.er[0] = -0.113917196281990;
  timectx.er[1] = 0.400761520311650;
  timectx.er[2] = -0.813632449486927;
  timectx.er[3] = 1.526788125457267;

  PetscScalar D1_time[4][4] = {
    {-3.3320002363522817*2./timectx.Tpb,4.8601544156851962*2./timectx.Tpb,-2.1087823484951789*2./timectx.Tpb,0.5806281691622644*2./timectx.Tpb},
    {-0.7575576147992339*2./timectx.Tpb,-0.3844143922232086*2./timectx.Tpb,1.4706702312807167*2./timectx.Tpb,-0.3286982242582743*2./timectx.Tpb},
    {0.3286982242582743*2./timectx.Tpb,-1.4706702312807167*2./timectx.Tpb,0.3844143922232086*2./timectx.Tpb,0.7575576147992339*2./timectx.Tpb},
    {-0.5806281691622644*2./timectx.Tpb,2.1087823484951789*2./timectx.Tpb,-4.8601544156851962*2./timectx.Tpb,3.3320002363522817*2./timectx.Tpb}
  };

  PetscScalar HI_BL_time[4][4] = {
    {6.701306630115196*2./timectx.Tpb,-3.571157279331500*2./timectx.Tpb,1.759003615747388*2./timectx.Tpb,-0.500000000000000*2./timectx.Tpb},
    {-1.904858685372247*2./timectx.Tpb,1.015107998460294*2./timectx.Tpb,-0.500000000000000*2./timectx.Tpb,0.142125915923019*2./timectx.Tpb},
    {0.938254199681965*2./timectx.Tpb,-0.500000000000000*2./timectx.Tpb,0.246279214013876*2./timectx.Tpb,-0.070005317729047*2./timectx.Tpb},
    {-0.500000000000000*2./timectx.Tpb,0.266452311201742*2./timectx.Tpb,-0.131243331549891*2./timectx.Tpb,0.037306157410634*2./timectx.Tpb}
  };

  for (i = 0; i < 4; i++) {
    for (j = 0; j < 4; j++) {
      timectx.D[j][i] = D1_time[j][i] + tau*HI_BL_time[j][i];
    }
  }

  return 0;
}