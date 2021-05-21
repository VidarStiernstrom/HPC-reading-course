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

all: init $(BIN_PATH)adv_1D $(BIN_PATH)adv_2D $(BIN_PATH)ref_1D $(BIN_PATH)acowave_2D

# Link object files to create binaries in BIN_PATH/
$(BIN_PATH)acowave_2D: $(OBJ_PATH)acowave_2D.o $(OBJ_PATH)create_layout.o $(OBJ_PATH)IO_utils.o $(OBJ_PATH)scatter_ctx.o $(OBJ_PATH)ts_rk.o
	-${CXX} -o $@ $^ ${PETSC_SYS_LIB}

$(BIN_PATH)adv_2D: $(OBJ_PATH)adv_2D.o $(OBJ_PATH)create_layout.o $(OBJ_PATH)IO_utils.o $(OBJ_PATH)scatter_ctx.o $(OBJ_PATH)ts_rk.o
	-${CXX} -o $@ $^ ${PETSC_SYS_LIB}

$(BIN_PATH)adv_1D: $(OBJ_PATH)adv_1D.o $(OBJ_PATH)create_layout.o $(OBJ_PATH)IO_utils.o $(OBJ_PATH)scatter_ctx.o $(OBJ_PATH)ts_rk.o
	-${CXX} -o $@ $^ ${PETSC_SYS_LIB}
	
$(BIN_PATH)ref_1D: $(OBJ_PATH)ref_1D.o $(OBJ_PATH)create_layout.o $(OBJ_PATH)IO_utils.o $(OBJ_PATH)scatter_ctx.o $(OBJ_PATH)ts_rk.o
	-${CXX} -o $@ $^ ${PETSC_SYS_LIB}

# Compile object files to OBJ_PATH/
$(OBJ_PATH)acowave_2D.o: $(SRC_PATH)acowave_2D.cpp
	-${CXX} ${CXXFLAGS} -c $^ -o $@ 
	
$(OBJ_PATH)adv_1D.o: $(SRC_PATH)adv_1D.cpp
	-${CXX} ${CXXFLAGS} -c $^ -o $@ 

$(OBJ_PATH)adv_2D.o: $(SRC_PATH)adv_2D.cpp
	-${CXX} ${CXXFLAGS} -c $^ -o $@ 

$(OBJ_PATH)ref_1D.o: $(SRC_PATH)ref_1D.cpp
	-${CXX} ${CXXFLAGS} -c $^ -o $@ 

$(OBJ_PATH)create_layout.o: $(SRC_PATH)grids/create_layout.cpp
	-${CXX} ${CXXFLAGS} -c $^ -o $@ 

$(OBJ_PATH)IO_utils.o: $(SRC_PATH)IO_utils.cpp
	-${CXX} ${CXXFLAGS} -c $^ -o $@ 

$(OBJ_PATH)scatter_ctx.o: $(SRC_PATH)scatter_ctx.cpp
	-${CXX} ${CXXFLAGS} -c $^ -o $@ 

$(OBJ_PATH)ts_rk.o: $(SRC_PATH)time_stepping/ts_rk.cpp
	-${CXX} ${CXXFLAGS} -c $^ -o $@ 


#.PHONY : clean
init:
	mkdir -p $(BIN_PATH)
	mkdir -p $(OBJ_PATH)

clean::
	-rm -f $(BIN_PATH)*
	-rm -f $(OBJ_PATH)*