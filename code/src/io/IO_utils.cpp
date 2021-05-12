#include<petsc.h>
#include <filesystem>

PetscErrorCode write_vector_to_binary(const Vec& v, const std::string folder, const std::string file)
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

void print_usage(char* exec_name) {
	PetscPrintf(PETSC_COMM_WORLD,"------------------------------ USAGE ------------------------------\n");
	PetscPrintf(PETSC_COMM_WORLD,"\"%s Nx Ny Tend CFL \"\n",exec_name);
	PetscPrintf(PETSC_COMM_WORLD,"\n");
	PetscPrintf(PETSC_COMM_WORLD,"Nx:\t\tnumber of grid points in x-direction.\n");
	PetscPrintf(PETSC_COMM_WORLD,"Ny:\t\tnumber of grid points in y-direction.\n");
	PetscPrintf(PETSC_COMM_WORLD,"Tend:\t\tfinal time.\n");
	PetscPrintf(PETSC_COMM_WORLD,"CFL:\t\tCFL number, dt = CFL*min(dx).\n");
	PetscPrintf(PETSC_COMM_WORLD,"------------------------------ Example ------------------------------\n");
	PetscPrintf(PETSC_COMM_WORLD,"\"%s 101 101 1 0.1 0 1\"\n",exec_name);
	PetscPrintf(PETSC_COMM_WORLD,"\n");
}

int get_inputs(int argc, char *argv[], PetscInt *Nx, PetscInt *Ny, PetscScalar *n_steps, PetscScalar *CFL) {

	if (argc != 5) {
		PetscPrintf(PETSC_COMM_WORLD,"Error, wrong number of input arguments. Expected 4 arguments, got %d.\n",argc-1);
		print_usage(argv[0]);
		return -1;
	}

	*Nx = atoi(argv[1]);
	if (*Nx <= 0) {
		PetscPrintf(PETSC_COMM_WORLD, "Error, first argument wrong. Expected Nx > 0, got %d.\n",*Nx);
		print_usage(argv[0]);
		return -1;
	}

	*Ny = atoi(argv[2]);
	if (*Ny <= 0) {
		PetscPrintf(PETSC_COMM_WORLD, "Error, second argument wrong. Expected Ny > 0, got %d.\n",*Ny);
		print_usage(argv[0]);
		return -1;
	}

	// *Tend = atof(argv[3]);
	// if (*Tend <= 0) {
	// 	PetscPrintf(PETSC_COMM_WORLD, "Error, third argument wrong. Expected Tend > 0, got %f.\n",*Tend);
	// 	print_usage(argv[0]);
	// 	return -1;
	// }

  *n_steps = atof(argv[3]);
	if (*n_steps <= 0) {
		PetscPrintf(PETSC_COMM_WORLD, "Error, third argument wrong. Expected n_steps > 0, got %f.\n",*n_steps);
		print_usage(argv[0]);
		return -1;
	}

	*CFL = atof(argv[4]);
	if (*CFL <= 0) {
		PetscPrintf(PETSC_COMM_WORLD, "Error, fourth argument wrong. Expected CFL > 0, got %f.\n",*CFL);
		print_usage(argv[0]);
		return -1;
	}
	return 0;
}
