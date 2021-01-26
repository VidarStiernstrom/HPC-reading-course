SRC_PATH = src/
BIN_PATH = bin/
OBJ_PATH = obj/
INCLUDE_PATH = include/
EXTERNAL_PATH = external/

# External include paths
MDSPAN_INCLUDE = $(EXTERNAL_PATH)mdspan/include
PETSC_INCLUDE = ${PETSC_DIR}include
EXTERNAL_INCLUDE = $(MDSPAN_INCLUDE) ${PETSC_INCLUDE}

INCLUDES = $(INCLUDE_PATH) $(EXTERNAL_INCLUDE)

IFLAGS			= $(foreach d, $(INCLUDES), -I$d)
DEBUG_MODULES	= -Wall
COPTFLAGS		= -std=c++17 -O3 -march=native -mtune=native
CPPFLAGS		= -DVERSION=${PETSC_VERSION_NUM}

CXX 			= mpicc 
CXXFLAGS		= $(DEBUG_MODULES) $(COPTFLAGS) $(IFLAGS)

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

all: init $(BIN_PATH)main_adv $(BIN_PATH)main_ref $(BIN_PATH)main_aco

# Link object files to create binaries in BIN_PATH/

$(BIN_PATH)main_aco: $(OBJ_PATH)main_aco.o $(OBJ_PATH)create_layout.o $(OBJ_PATH)IO_utils.o $(OBJ_PATH)scatter_ctx.o $(OBJ_PATH)multigrid.o $(OBJ_PATH)standard.o $(OBJ_PATH)imp_timestepping.o $(OBJ_PATH)aco_2D.o
	-${CXX} -o $@ $^ ${PETSC_SYS_LIB}

$(BIN_PATH)main_adv: $(OBJ_PATH)main_adv.o $(OBJ_PATH)create_layout.o $(OBJ_PATH)IO_utils.o $(OBJ_PATH)scatter_ctx.o $(OBJ_PATH)multigrid.o $(OBJ_PATH)standard.o $(OBJ_PATH)imp_timestepping.o $(OBJ_PATH)adv_1D.o
	-${CXX} -o $@ $^ ${PETSC_SYS_LIB}
	
$(BIN_PATH)main_ref: $(OBJ_PATH)main_ref.o $(OBJ_PATH)create_layout.o $(OBJ_PATH)IO_utils.o $(OBJ_PATH)scatter_ctx.o $(OBJ_PATH)multigrid.o $(OBJ_PATH)standard.o $(OBJ_PATH)imp_timestepping.o $(OBJ_PATH)ref_1D.o
	-${CXX} -o $@ $^ ${PETSC_SYS_LIB}

# Compile object files to OBJ_PATH/
$(OBJ_PATH)main_aco.o: $(SRC_PATH)main_aco.cpp
	-${CXX} ${CXXFLAGS} -c $^ -o $@ 
	
$(OBJ_PATH)main_adv.o: $(SRC_PATH)main_adv.cpp
	-${CXX} ${CXXFLAGS} -c $^ -o $@ 

$(OBJ_PATH)main_ref.o: $(SRC_PATH)main_ref.cpp
	-${CXX} ${CXXFLAGS} -c $^ -o $@ 

$(OBJ_PATH)create_layout.o: $(SRC_PATH)grids/create_layout.cpp
	-${CXX} ${CXXFLAGS} -c $^ -o $@ 

$(OBJ_PATH)IO_utils.o: $(SRC_PATH)IO_utils.cpp
	-${CXX} ${CXXFLAGS} -c $^ -o $@ 

$(OBJ_PATH)scatter_ctx.o: $(SRC_PATH)scatter_ctx.cpp
	-${CXX} ${CXXFLAGS} -c $^ -o $@ 

$(OBJ_PATH)multigrid.o: $(SRC_PATH)multigrid.cpp
	-${CXX} ${CXXFLAGS} -c $^ -o $@ 

$(OBJ_PATH)standard.o: $(SRC_PATH)standard.cpp
	-${CXX} ${CXXFLAGS} -c $^ -o $@ 

$(OBJ_PATH)imp_timestepping.o: $(SRC_PATH)imp_timestepping.cpp
	-${CXX} ${CXXFLAGS} -c $^ -o $@ 

$(OBJ_PATH)adv_1D.o: $(SRC_PATH)adv_1D.cpp
	-${CXX} ${CXXFLAGS} -c $^ -o $@ 

$(OBJ_PATH)ref_1D.o: $(SRC_PATH)ref_1D.cpp
	-${CXX} ${CXXFLAGS} -c $^ -o $@ 

$(OBJ_PATH)aco_2D.o: $(SRC_PATH)aco_2D.cpp
	-${CXX} ${CXXFLAGS} -c $^ -o $@ 


#.PHONY : clean
init:
	mkdir -p $(BIN_PATH)
	mkdir -p $(OBJ_PATH)

clean::
	-rm -f $(BIN_PATH)*
	-rm -f $(OBJ_PATH)*