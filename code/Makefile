SRC_PATH = src/
BIN_PATH = bin/
INCLUDE_PATH = include/

all: $(BIN_PATH)adv_1D $(BIN_PATH)adv_2D $(BIN_PATH)ref_1D $(BIN_PATH)sim_adv_ts_old

IFLAGS			= -I$(INCLUDE_PATH)
DEBUG_MODULES	= -Wall
COPTFLAGS		= -std=c++17 -O3 -march=native -mtune=native
CPPFLAGS		= -DVERSION=${PETSC_VERSION_NUM}
CXXFLAGS		= $(DEBUG_MODULES) $(COPTFLAGS) $(IFLAGS)
CLINKER			= mpicc


include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

$(BIN_PATH)adv_2D: $(SRC_PATH)adv_2D.o
	mkdir -p $(BIN_PATH)
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}
	mv $(SRC_PATH)*.o $(BIN_PATH)

$(BIN_PATH)adv_1D: $(SRC_PATH)adv_1D.o
	mkdir -p $(BIN_PATH)
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}
	mv $(SRC_PATH)*.o $(BIN_PATH)
	
$(BIN_PATH)ref_1D: $(SRC_PATH)ref_1D.o
	mkdir -p $(BIN_PATH)
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}
	mv $(SRC_PATH)*.o $(BIN_PATH)

$(BIN_PATH)sim_adv_ts_old: $(SRC_PATH)sim_adv_ts_old.o
	mkdir -p $(BIN_PATH)
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}
	mv $(SRC_PATH)*.o $(BIN_PATH)

#.PHONY : clean
clean::
	-rm -f $(BIN_PATH)*.o $(BIN_PATH)adv_1D
	-rm -f $(BIN_PATH)*.o $(BIN_PATH)adv_2D
	-rm -f $(BIN_PATH)*.o $(BIN_PATH)ref_1D
	-rm -f $(BIN_PATH)*.o $(BIN_PATH)sim_adv_ts