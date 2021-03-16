SRC_PATH = src/
BIN_PATH = bin/
OBJ_PATH = obj/
INCLUDE_PATH = include/
EXTERNAL_PATH = external/

all: init $(BIN_PATH)acoustic_wave_eq

# External include paths
MDSPAN_INCLUDE = $(EXTERNAL_PATH)mdspan/include
PETSC_INCLUDE = ${PETSC_DIR}include
EXTERNAL_INCLUDE = $(MDSPAN_INCLUDE) ${PETSC_INCLUDE}

INCLUDES = $(INCLUDE_PATH) $(EXTERNAL_INCLUDE)

IFLAGS			= $(foreach d, $(INCLUDES), -I$d)
DEBUG_MODULES	= -Wall
COPTFLAGS		= -std=c++17 -O3 -march=native -mtune=native -ffast-math
CPPFLAGS		= -DVERSION=${PETSC_VERSION_NUM}

CXX 			= mpicc 
CXXFLAGS		= $(DEBUG_MODULES) $(COPTFLAGS) $(IFLAGS)

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

# Link object files to create binaries in BIN_PATH/
$(BIN_PATH)acoustic_wave_eq: $(OBJ_PATH)acoustic_wave_eq.o $(OBJ_PATH)create_layout.o $(OBJ_PATH)IO_utils.o $(OBJ_PATH)scatter_ctx.o $(OBJ_PATH)timestepping.o
	-${CXX} -o $@ $^ ${PETSC_SYS_LIB}

# Compile object files to OBJ_PATH/
$(OBJ_PATH)acoustic_wave_eq.o: $(SRC_PATH)acoustic_wave_eq/acowave_2D.cpp
	-${CXX} ${CXXFLAGS} -c $^ -o $@ 
	
$(OBJ_PATH)create_layout.o: $(SRC_PATH)grids/create_layout.cpp
	-${CXX} ${CXXFLAGS} -c $^ -o $@ 

$(OBJ_PATH)IO_utils.o: $(SRC_PATH)io/IO_utils.cpp
	-${CXX} ${CXXFLAGS} -c $^ -o $@ 

$(OBJ_PATH)scatter_ctx.o: $(SRC_PATH)scatter_ctx/scatter_ctx.cpp
	-${CXX} ${CXXFLAGS} -c $^ -o $@ 

$(OBJ_PATH)timestepping.o: $(SRC_PATH)timestepping/timestepping.cpp
	-${CXX} ${CXXFLAGS} -c $^ -o $@ 


#.PHONY : clean
init:
	mkdir -p $(BIN_PATH)
	mkdir -p $(OBJ_PATH)

clean::
	-rm -f $(BIN_PATH)*
	-rm -f $(OBJ_PATH)*