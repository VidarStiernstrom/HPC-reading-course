all: debug

SRC_PATH = src/
BIN_PATH = bin/
OBJ_PATH = obj/
PROF_PATH = profiling/
INCLUDE_PATH = include/
EXTERNAL_PATH = external/

# External include paths
MDSPAN_INCLUDE = $(EXTERNAL_PATH)mdspan/include
PETSC_INCLUDE = ${PETSC_DIR}/include
EXTERNAL_INCLUDE = $(MDSPAN_INCLUDE) ${PETSC_INCLUDE}

INCLUDES = $(INCLUDE_PATH) $(EXTERNAL_INCLUDE)

IFLAGS			= $(foreach d, $(INCLUDES), -I$d)
DEBUGFLAGS		= -Wall -g 
COPTFLAGS		= -O3 -march=native -mtune=native -ffast-math
CPPFLAGS		= -DVERSION=${PETSC_VERSION_NUM}
LDFLAGS			= -fprofile-arcs #-fprofile-dir=/$(PROF_PATH)
GCOV	    	= -fprofile-arcs -ftest-coverage #-fprofile-dir=/$(PROF_PATH)
GPROF			= -pg -fprofile-dir=/$(PROF_PATH)
PROFILING		= false
COVERAGE		= false

ifeq ($(strip $(ORDER)),)
ORDER			= 4
ORDER_MSG		= ORDER not set. Compiling acoustic_wave_sim with $(ORDER)
else
ORDER_MSG		= Compiling acoustic_wave_sim with $(ORDER)
endif
CXX 			= mpicc 
CXXFLAGS		= -std=c++17 $(IFLAGS)

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

debug: CXXFLAGS += -DDEBUG $(DEBUGFLAGS)
debug: init acoustic_wave_sim
opt: CXXFLAGS += -DNDEBUG $(COPTFLAGS)
opt: init acoustic_wave_sim

# Link object files to create binaries in BIN_PATH/
acoustic_wave_sim: acoustic_wave_sim.o IO_utils.o timestepping.o
ifeq (${PROFILING}, true)
	-${CXX} -o $@ ${GPROF} $(OBJ_PATH)acoustic_wave_sim.o $(OBJ_PATH)IO_utils.o $(OBJ_PATH)timestepping.o  ${PETSC_SYS_LIB}
	mv acoustic_wave_sim $(BIN_PATH)acoustic_wave_sim
else
ifeq (${COVERAGE}, true)
	-${CXX} -o $@ ${GCOV} $(OBJ_PATH)acoustic_wave_sim.o $(OBJ_PATH)IO_utils.o $(OBJ_PATH)timestepping.o  ${PETSC_SYS_LIB}
	mv acoustic_wave_sim $(BIN_PATH)acoustic_wave_sim
else
	-${CXX} -o $@ $(OBJ_PATH)acoustic_wave_sim.o $(OBJ_PATH)IO_utils.o $(OBJ_PATH)timestepping.o  ${PETSC_SYS_LIB}
	mv acoustic_wave_sim $(BIN_PATH)acoustic_wave_sim
endif
endif

# Compile object files to OBJ_PATH/
acoustic_wave_sim.o: $(SRC_PATH)acoustic_wave_sim/acoustic_wave_sim.cpp $(INCLUDE_PATH)acoustic_wave_eq/wave_eq_rhs.h $(INCLUDE_PATH)$(wildcard sbpops/*.h)
ifeq (${PROFILING}, true)
	echo $(ORDER_MSG)
	-${CXX} ${CXXFLAGS} ${GPROF} -c $(SRC_PATH)acoustic_wave_sim/acoustic_wave_sim.cpp -DSBP_OPERATOR_ORDER=$(ORDER)
	mv acoustic_wave_sim.o $(OBJ_PATH)acoustic_wave_sim.o
else
ifeq (${COVERAGE}, true)
	echo $(ORDER_MSG)
	-${CXX} ${CXXFLAGS} ${GCOV} -c $(SRC_PATH)acoustic_wave_sim/acoustic_wave_sim.cpp -DSBP_OPERATOR_ORDER=$(ORDER)
	mv acoustic_wave_sim.o $(OBJ_PATH)acoustic_wave_sim.o
else
	echo $(ORDER_MSG)
	-${CXX} ${CXXFLAGS} -c $(SRC_PATH)acoustic_wave_sim/acoustic_wave_sim.cpp -DSBP_OPERATOR_ORDER=$(ORDER)
	mv acoustic_wave_sim.o $(OBJ_PATH)acoustic_wave_sim.o
endif
endif


create_layout.o: $(SRC_PATH)grids/create_layout.cpp  $(INCLUDE_PATH)/grids/create_layout.h $(INCLUDE_PATH)/grids/layout.h
ifeq (${PROFILING}, true)
	-${CXX} ${CXXFLAGS} ${GPROF} -c $(SRC_PATH)grids/create_layout.cpp
	mv create_layout.o $(OBJ_PATH)create_layout.o
else
ifeq (${COVERAGE}, true)
	-${CXX} ${CXXFLAGS} ${GCOV} -c $(SRC_PATH)grids/create_layout.cpp
	mv create_layout.o $(OBJ_PATH)create_layout.o
else
	-${CXX} ${CXXFLAGS} -c $(SRC_PATH)grids/create_layout.cpp
	mv create_layout.o $(OBJ_PATH)create_layout.o
endif
endif


IO_utils.o: $(SRC_PATH)io/IO_utils.cpp $(INCLUDE_PATH)/io/IO_utils.h
ifeq (${PROFILING}, true)
	-${CXX} ${CXXFLAGS} ${GPROF} -c $(SRC_PATH)io/IO_utils.cpp
	mv IO_utils.o $(OBJ_PATH)IO_utils.o
else
ifeq (${COVERAGE}, true)
	-${CXX} ${CXXFLAGS} ${GCOV} -c $(SRC_PATH)io/IO_utils.cpp
	mv IO_utils.o $(OBJ_PATH)IO_utils.o
else
	-${CXX} ${CXXFLAGS} -c $(SRC_PATH)io/IO_utils.cpp
	mv IO_utils.o $(OBJ_PATH)IO_utils.o
endif
endif


scatter_ctx.o: $(SRC_PATH)scatter_ctx/scatter_ctx.cpp $(INCLUDE_PATH)/scatter_ctx/scatter_ctx.h
ifeq (${PROFILING}, true)
	-${CXX} ${CXXFLAGS} ${GPROF} -c $(SRC_PATH)scatter_ctx/scatter_ctx.cpp
	mv scatter_ctx.o $(OBJ_PATH)scatter_ctx.o
else
ifeq (${COVERAGE}, true)
	-${CXX} ${CXXFLAGS} $(GCOV) -c $(SRC_PATH)scatter_ctx/scatter_ctx.cpp
	mv scatter_ctx.o $(OBJ_PATH)scatter_ctx.o
else
	-${CXX} ${CXXFLAGS} -c $(SRC_PATH)scatter_ctx/scatter_ctx.cpp
	mv scatter_ctx.o $(OBJ_PATH)scatter_ctx.o
endif
endif


timestepping.o: $(SRC_PATH)timestepping/timestepping.cpp $(INCLUDE_PATH)/timestepping/timestepping.h
ifeq (${PROFILING}, true)
	-${CXX} ${CXXFLAGS} ${GPROF} -c $(SRC_PATH)timestepping/timestepping.cpp
	mv timestepping.o $(OBJ_PATH)timestepping.o
else
ifeq (${COVERAGE}, true)
	-${CXX} ${CXXFLAGS} $(GCOV) -c $(SRC_PATH)timestepping/timestepping.cpp
	mv timestepping.o $(OBJ_PATH)timestepping.o
else
	-${CXX} ${CXXFLAGS} -c $(SRC_PATH)timestepping/timestepping.cpp
	mv timestepping.o $(OBJ_PATH)timestepping.o
endif
endif


#.PHONY : clean
init:
	mkdir -p $(BIN_PATH)
	mkdir -p $(OBJ_PATH)
	mkdir -p $(PROF_PATH)

clean::
	-rm -f $(BIN_PATH)*
	-rm -f $(OBJ_PATH)*
	-rm -f *.gcno *.gcda *.gcov gmon.out.*
