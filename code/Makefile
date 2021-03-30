SRC_PATH = src/
BIN_PATH = bin/
OBJ_PATH = obj/
INCLUDE_PATH = include/
EXTERNAL_PATH = external/

all: init acoustic_wave_sim

# External include paths
MDSPAN_INCLUDE = $(EXTERNAL_PATH)mdspan/include
PETSC_INCLUDE = ${PETSC_DIR}include
EXTERNAL_INCLUDE = $(MDSPAN_INCLUDE) ${PETSC_INCLUDE}

INCLUDES = $(INCLUDE_PATH) $(EXTERNAL_INCLUDE)

IFLAGS			= $(foreach d, $(INCLUDES), -I$d)
DEBUG_MODULES	= -Wall
COPTFLAGS		= -std=c++17 -O3 -march=native -mtune=native -ffast-math
CPPFLAGS		= -DVERSION=${PETSC_VERSION_NUM}

ifeq ($(strip $(ORDER)),)
ORDER		= 4
ORDER_MSG	= ORDER not set. Using ORDER=$(ORDER)
else
ORDER_MSG	= Compiling acoustic_wave_sim with $(ORDER)
endif
CXX 			= mpicc 
CXXFLAGS		= $(DEBUG_MODULES) $(COPTFLAGS) $(IFLAGS)

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

# Link object files to create binaries in BIN_PATH/
acoustic_wave_sim: acoustic_wave_sim.o IO_utils.o timestepping.o 
	-${CXX} -o $@ $(OBJ_PATH)acoustic_wave_sim.o $(OBJ_PATH)IO_utils.o $(OBJ_PATH)timestepping.o  ${PETSC_SYS_LIB}
	mv acoustic_wave_sim $(BIN_PATH)acoustic_wave_sim

# Compile object files to OBJ_PATH/
acoustic_wave_sim.o: $(SRC_PATH)acoustic_wave_sim/acoustic_wave_sim.cpp $(INCLUDE_PATH)acoustic_wave_eq/wave_eq_rhs.h $(INCLUDE_PATH)$(wildcard sbpops/*.h)
	echo $(ORDER_MSG)
	-${CXX} ${CXXFLAGS} -c $(SRC_PATH)acoustic_wave_sim/acoustic_wave_sim.cpp -DSBP_OPERATOR_ORDER=$(ORDER)
	mv acoustic_wave_sim.o $(OBJ_PATH)acoustic_wave_sim.o
	
create_layout.o: $(SRC_PATH)grids/create_layout.cpp  $(INCLUDE_PATH)/grids/create_layout.h $(INCLUDE_PATH)/grids/layout.h
	-${CXX} ${CXXFLAGS} -c $(SRC_PATH)grids/create_layout.cpp
	mv create_layout.o $(OBJ_PATH)create_layout.o

IO_utils.o: $(SRC_PATH)io/IO_utils.cpp $(INCLUDE_PATH)/io/IO_utils.h
	-${CXX} ${CXXFLAGS} -c $(SRC_PATH)io/IO_utils.cpp
	mv IO_utils.o $(OBJ_PATH)IO_utils.o

scatter_ctx.o: $(SRC_PATH)scatter_ctx/scatter_ctx.cpp $(INCLUDE_PATH)/scatter_ctx/scatter_ctx.h
	-${CXX} ${CXXFLAGS} -c $(SRC_PATH)scatter_ctx/scatter_ctx.cpp
	mv scatter_ctx.o $(OBJ_PATH)scatter_ctx.o

timestepping.o: $(SRC_PATH)timestepping/timestepping.cpp $(INCLUDE_PATH)/timestepping/timestepping.h
	-${CXX} ${CXXFLAGS} -c $(SRC_PATH)timestepping/timestepping.cpp
	mv timestepping.o $(OBJ_PATH)timestepping.o


#.PHONY : clean
init:
	mkdir -p $(BIN_PATH)
	mkdir -p $(OBJ_PATH)

clean::
	-rm -f $(BIN_PATH)*
	-rm -f $(OBJ_PATH)*