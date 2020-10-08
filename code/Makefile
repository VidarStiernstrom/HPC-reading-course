SRC_PATH = src/
BIN_PATH = bin/
OBJ_PATH = obj/
INCLUDE_PATH = include/

IFLAGS			= -I$(INCLUDE_PATH) -I${PETSC_DIR}/include
DEBUG_MODULES	= -Wall
COPTFLAGS		= -std=c++17 -O3 -march=native -mtune=native
CPPFLAGS		= -DVERSION=${PETSC_VERSION_NUM}

CXX 			= mpicc 
CXXFLAGS		= $(DEBUG_MODULES) $(COPTFLAGS) $(IFLAGS)

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

all: init $(BIN_PATH)adv_1D $(BIN_PATH)adv_2D $(BIN_PATH)ref_1D $(BIN_PATH)ref_2D $(BIN_PATH)sim_adv_ts_old

# Link object files to create binaries in BIN_PATH/
$(BIN_PATH)adv_2D: $(OBJ_PATH)adv_2D.o
	-${CXX} -o $@ $^ ${PETSC_SYS_LIB}

$(BIN_PATH)adv_1D: $(OBJ_PATH)adv_1D.o $(OBJ_PATH)create_layout.o 
	-${CXX} -o $@ $^ ${PETSC_SYS_LIB}
	
$(BIN_PATH)ref_1D: $(OBJ_PATH)ref_1D.o $(OBJ_PATH)create_layout.o 
	-${CXX} -o $@ $^ ${PETSC_SYS_LIB}

$(BIN_PATH)ref_2D: $(OBJ_PATH)ref_2D.o
	-${CXX} -o $@ $^ ${PETSC_SYS_LIB}

$(BIN_PATH)sim_adv_ts_old: $(OBJ_PATH)sim_adv_ts_old.o
	-${CXX} -o $@ $^ ${PETSC_SYS_LIB}

# Compile object files to OBJ_PATH/
$(OBJ_PATH)adv_1D.o: $(SRC_PATH)adv_1D.cpp
	-${CXX} ${CXXFLAGS} -c $^ -o $@ 

$(OBJ_PATH)adv_2D.o: $(SRC_PATH)adv_2D.cpp
	-${CXX} ${CXXFLAGS} -c $^ -o $@ 

$(OBJ_PATH)ref_1D.o: $(SRC_PATH)ref_1D.cpp
	-${CXX} ${CXXFLAGS} -c $^ -o $@ 

$(OBJ_PATH)ref_2D.o: $(SRC_PATH)ref_2D.cpp
	-${CXX} ${CXXFLAGS} -c $^ -o $@ 

$(OBJ_PATH)create_layout.o: $(SRC_PATH)grids/create_layout.cpp
	-${CXX} ${CXXFLAGS} -c $^ -o $@ 


#.PHONY : clean
init:
	mkdir -p $(BIN_PATH)
	mkdir -p $(OBJ_PATH)

clean::
	-rm -f $(BIN_PATH)*
	-rm -f $(OBJ_PATH)*