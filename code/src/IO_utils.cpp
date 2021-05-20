#include<petsc.h>
#include <filesystem>

PetscErrorCode write_vector_to_binary(const Vec v, const std::string folder, const std::string file)
{ 
  std::filesystem::create_directories(folder);
  PetscErrorCode ierr;
  PetscViewer viewer;
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(folder+"/"+file).c_str(),FILE_MODE_WRITE,&viewer);
  ierr = VecView(v,viewer);
  ierr = PetscViewerDestroy(&viewer);
  CHKERRQ(ierr);
  return 0;
}

PetscErrorCode write_data_to_file(const std::string data_string, const std::string folder, const std::string file)
{ 
  PetscInt rank;

  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  std::filesystem::create_directories(folder);
  if (rank == 0) {
    FILE *f = fopen((folder+"/"+file).c_str(), "a");
    if (!f) {
      printf("File '%s' failed to open.\n",(folder+"/"+file).c_str());
      return -1;
    }
    fseek(f, 0, SEEK_END);
    if (!ftell(f)) {
      fprintf(f, "Size\tNx\tNy\tdt\tTend\telapsed_time\tl2-error\tH-error\tmax-error\n");
    }
    fprintf(f,"%s",data_string.c_str());
    fclose(f);
  }
  return 0;
}

void print_usage_2d(char* exec_name) {
	PetscPrintf(PETSC_COMM_WORLD,"------------------------------ USAGE ------------------------------\n");
	PetscPrintf(PETSC_COMM_WORLD,"\"%s Nx Ny Tend CFL use_custom_ts use_custom_sc\"\n",exec_name);
	PetscPrintf(PETSC_COMM_WORLD,"\n");
	PetscPrintf(PETSC_COMM_WORLD,"Nx:\t\tnumber of grid points in x-direction.\n");
	PetscPrintf(PETSC_COMM_WORLD,"Ny:\t\tnumber of grid points in y-direction.\n");
	PetscPrintf(PETSC_COMM_WORLD,"Tend:\t\tfinal time.\n");
	PetscPrintf(PETSC_COMM_WORLD,"CFL:\t\tCFL number, dt = CFL*min(dx).\n");
	PetscPrintf(PETSC_COMM_WORLD,"use_custom_ts:\t1 - use custom time stepper, 0 - use PETSc time stepper.\n");
	PetscPrintf(PETSC_COMM_WORLD,"use_custom_sc:\t1 - use custom scatter context, 0 - use PETSc scatter context.\n");
	PetscPrintf(PETSC_COMM_WORLD,"------------------------------ Example ------------------------------\n");
	PetscPrintf(PETSC_COMM_WORLD,"\"%s 101 101 1 0.1 0 1\"\n",exec_name);
	PetscPrintf(PETSC_COMM_WORLD,"\n");
}

int get_inputs_2d(int argc, char *argv[], PetscInt *Nx, PetscInt *Ny, PetscScalar *Tend, PetscScalar *CFL, PetscBool *use_custom_ts, PetscBool *use_custom_sc) {

	if (argc != 7) {
		PetscPrintf(PETSC_COMM_WORLD,"Error, wrong number of input arguments. Expected 6 arguments, got %d.\n",argc-1);
		print_usage_2d(argv[0]);
		return -1;
	}

	*Nx = atoi(argv[1]);
	if (*Nx <= 0) {
		PetscPrintf(PETSC_COMM_WORLD, "Error, first argument wrong. Expected Nx > 0, got %d.\n",*Nx);
		print_usage_2d(argv[0]);
		return -1;
	}

	*Ny = atoi(argv[2]);
	if (*Ny <= 0) {
		PetscPrintf(PETSC_COMM_WORLD, "Error, second argument wrong. Expected Ny > 0, got %d.\n",*Ny);
		print_usage_2d(argv[0]);
		return -1;
	}

	*Tend = atof(argv[3]);
	if (*Tend <= 0) {
		PetscPrintf(PETSC_COMM_WORLD, "Error, third argument wrong. Expected Tend > 0, got %f.\n",*Tend);
		print_usage_2d(argv[0]);
		return -1;
	}

	*CFL = atof(argv[4]);
	if (*CFL <= 0) {
		PetscPrintf(PETSC_COMM_WORLD, "Error, fourth argument wrong. Expected CFL > 0, got %f.\n",*CFL);
		print_usage_2d(argv[0]);
		return -1;
	}

	*use_custom_ts = (PetscBool) atoi(argv[5]);
	if (*use_custom_ts != 0 && *use_custom_ts != 1) {
		PetscPrintf(PETSC_COMM_WORLD, "Error, fifth argument wrong. Expected use_custom_ts = 1 or 0, got %d.\n",*use_custom_ts);
		print_usage_2d(argv[0]);
		return -1;
	}

	*use_custom_sc = (PetscBool) atoi(argv[6]);
	if (*use_custom_sc != 0 && *use_custom_sc != 1) {
		PetscPrintf(PETSC_COMM_WORLD, "Error, sixth argument wrong. Expected use_custom_sc = 1 or 0, got %d.\n",*use_custom_sc);
		print_usage_2d(argv[0]);
		return -1;
	}

	return 0;
}

void print_usage_1d(char* exec_name) {
	PetscPrintf(PETSC_COMM_WORLD,"------------------------------ USAGE ------------------------------\n");
	PetscPrintf(PETSC_COMM_WORLD,"\"%s N Tend CFL use_custom_ts use_custom_sc\"\n",exec_name);
	PetscPrintf(PETSC_COMM_WORLD,"\n");
	PetscPrintf(PETSC_COMM_WORLD,"N:\t\tnumber of grid points.\n");
	PetscPrintf(PETSC_COMM_WORLD,"Tend:\t\tfinal time.\n");
	PetscPrintf(PETSC_COMM_WORLD,"CFL:\t\tCFL number, dt = CFL*min(dx).\n");
	PetscPrintf(PETSC_COMM_WORLD,"use_custom_ts:\t1 - use custom time stepper, 0 - use PETSc time stepper.\n");
	PetscPrintf(PETSC_COMM_WORLD,"use_custom_sc:\t1 - use custom scatter context, 0 - use PETSc scatter context.\n");
	PetscPrintf(PETSC_COMM_WORLD,"------------------------------ Example ------------------------------\n");
	PetscPrintf(PETSC_COMM_WORLD,"\"%s 101 1 0.1 0 1\"\n",exec_name);
	PetscPrintf(PETSC_COMM_WORLD,"\n");
}

int get_inputs_1d(int argc, char *argv[], PetscInt *N, PetscScalar *Tend, PetscScalar *CFL, PetscBool *use_custom_ts, PetscBool *use_custom_sc) {

	if (argc != 6) {
		PetscPrintf(PETSC_COMM_WORLD,"Error, wrong number of input arguments. Expected 5 arguments, got %d.\n",argc-1);
		print_usage_1d(argv[0]);
		return -1;
	}

	*N = atoi(argv[1]);
	if (*N <= 0) {
		PetscPrintf(PETSC_COMM_WORLD, "Error, second argument wrong. Expected N > 0, got %d.\n",*N);
		print_usage_1d(argv[0]);
		return -1;
	}

	*Tend = atof(argv[2]);
	if (*Tend <= 0) {
		PetscPrintf(PETSC_COMM_WORLD, "Error, third argument wrong. Expected Tend > 0, got %f.\n",*Tend);
		print_usage_1d(argv[0]);
		return -1;
	}

	*CFL = atof(argv[3]);
	if (*CFL <= 0) {
		PetscPrintf(PETSC_COMM_WORLD, "Error, fourth argument wrong. Expected CFL > 0, got %f.\n",*CFL);
		print_usage_1d(argv[0]);
		return -1;
	}

	*use_custom_ts = (PetscBool) atoi(argv[4]);
	if (*use_custom_ts != 0 && *use_custom_ts != 1) {
		PetscPrintf(PETSC_COMM_WORLD, "Error, fifth argument wrong. Expected use_custom_ts = 1 or 0, got %d.\n",*use_custom_ts);
		print_usage_1d(argv[0]);
		return -1;
	}

	*use_custom_sc = (PetscBool) atoi(argv[5]);
	if (*use_custom_sc != 0 && *use_custom_sc != 1) {
		PetscPrintf(PETSC_COMM_WORLD, "Error, sixth argument wrong. Expected use_custom_sc = 1 or 0, got %d.\n",*use_custom_sc);
		print_usage_1d(argv[0]);
		return -1;
	}

	return 0;
}