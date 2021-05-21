SRC_PATH = src
BIN_PATH = bin
OBJ_PATH = obj
INCLUDE_PATH = include
EXTERNAL_PATH = external

# External include paths
MDSPAN_INCLUDE = $(EXTERNAL_PATH)/mdspan/include
PETSC_INCLUDE = ${PETSC_DIR}include
EXTERNAL_INCLUDE = $(MDSPAN_INCLUDE) ${PETSC_INCLUDE}

INCLUDES = $(INCLUDE_PATH) $(EXTERNAL_INCLUDE)

IFLAGS			= $(foreach d, $(INCLUDES), -I$d)
DEBUGFLAGS		= -Wall -g
COPTFLAGS		= -O3 -march=native -mtune=native -ffast-math
CPPFLAGS		= -DVERSION=${PETSC_VERSION_NUM}

ifeq ($(strip $(order)),)
order		= 4
ORDER_MSG	= order not set. Compiling with $(order)
else
ORDER_MSG	= Compiling with $(order)
endif
CXX 			= mpicc 
CXXFLAGS		= -std=c++17 $(IFLAGS)

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

opt: CXXFLAGS += -DNDEBUG $(COPTFLAGS)
opt: $(app)
debug: CXXFLAGS += -DDEBUG $(DEBUGFLAGS)
debug: $(app)	
opt-debug: CXXFLAGS += -DDEBUG $(DEBUGFLAGS) $(COPTFLAGS)
opt-debug: $(app)
all: acowave_2D adv_2D adv_1D ref_1D

# Link object files to create binaries in BIN_PATH/
acowave_2D: acowave_2D.o io_utils.o ts_rk.o scatter_ctx.o create_layout.o
	-${CXX} -o $@ $(OBJ_PATH)/acowave_2D.o $(OBJ_PATH)/io_utils.o  $(OBJ_PATH)/scatter_ctx.o $(OBJ_PATH)/create_layout.o $(OBJ_PATH)/ts_rk.o  ${PETSC_SYS_LIB}
	mv acowave_2D $(BIN_PATH)/acowave_2D

adv_2D: adv_2D.o io_utils.o ts_rk.o scatter_ctx.o create_layout.o
	-${CXX} -o $@ $(OBJ_PATH)/adv_2D.o $(OBJ_PATH)/io_utils.o  $(OBJ_PATH)/scatter_ctx.o $(OBJ_PATH)/create_layout.o $(OBJ_PATH)/ts_rk.o  ${PETSC_SYS_LIB}
	mv adv_2D $(BIN_PATH)/adv_2D

adv_1D: adv_1D.o io_utils.o ts_rk.o scatter_ctx.o create_layout.o
	-${CXX} -o $@ $(OBJ_PATH)/adv_1D.o $(OBJ_PATH)/io_utils.o  $(OBJ_PATH)/scatter_ctx.o $(OBJ_PATH)/create_layout.o $(OBJ_PATH)/ts_rk.o ${PETSC_SYS_LIB}
	mv adv_1D $(BIN_PATH)/adv_1D

ref_1D: ref_1D.o io_utils.o ts_rk.o scatter_ctx.o create_layout.o
	-${CXX} -o $@ $(OBJ_PATH)/ref_1D.o $(OBJ_PATH)/io_utils.o  $(OBJ_PATH)/scatter_ctx.o $(OBJ_PATH)/create_layout.o $(OBJ_PATH)/ts_rk.o ${PETSC_SYS_LIB}
	mv ref_1D $(BIN_PATH)/ref_1D

# Compile object files to OBJ_PATH/
acowave_2D.o: $(SRC_PATH)/acowave_2D.cpp $(INCLUDE_PATH)/diffops/wave_eq.h $(INCLUDE_PATH)/$(wildcard sbpops/*.h)
	echo $(ORDER_MSG)
	-${CXX} ${CXXFLAGS} -c $(SRC_PATH)/acowave_2D.cpp -DSBP_OPERATOR_ORDER=$(order)
	mv acowave_2D.o $(OBJ_PATH)/acowave_2D.o

adv_2D.o: $(SRC_PATH)/adv_2D.cpp $(INCLUDE_PATH)/diffops/advection.h $(INCLUDE_PATH)/$(wildcard sbpops/*.h)
	echo $(ORDER_MSG)
	-${CXX} ${CXXFLAGS} -c $(SRC_PATH)/adv_2D.cpp -DSBP_OPERATOR_ORDER=$(order)
	mv adv_2D.o $(OBJ_PATH)/adv_2D.o

adv_1D.o: $(SRC_PATH)/adv_1D.cpp $(INCLUDE_PATH)/diffops/advection.h $(INCLUDE_PATH)/$(wildcard sbpops/*.h)
	echo $(ORDER_MSG)
	-${CXX} ${CXXFLAGS} -c $(SRC_PATH)/adv_1D.cpp -DSBP_OPERATOR_ORDER=$(order)
	mv adv_1D.o $(OBJ_PATH)/adv_1D.o

ref_1D.o: $(SRC_PATH)/ref_1D.cpp $(INCLUDE_PATH)/diffops/reflection.h $(INCLUDE_PATH)/$(wildcard sbpops/*.h)
	echo $(ORDER_MSG)
	-${CXX} ${CXXFLAGS} -c $(SRC_PATH)/ref_1D.cpp -DSBP_OPERATOR_ORDER=$(order)
	mv ref_1D.o $(OBJ_PATH)/ref_1D.o	
	
create_layout.o: $(SRC_PATH)/grids/create_layout.cpp  $(INCLUDE_PATH)/grids/create_layout.h $(INCLUDE_PATH)/grids/layout.h
	-${CXX} ${CXXFLAGS} -c $(SRC_PATH)/grids/create_layout.cpp
	mv create_layout.o $(OBJ_PATH)/create_layout.o

io_utils.o: $(SRC_PATH)/io/io_utils.cpp $(INCLUDE_PATH)/io/io_utils.h
	-${CXX} ${CXXFLAGS} -c $(SRC_PATH)/io/io_utils.cpp
	mv io_utils.o $(OBJ_PATH)/io_utils.o

scatter_ctx.o: $(SRC_PATH)/scatter_ctx/scatter_ctx.cpp $(INCLUDE_PATH)/scatter_ctx/scatter_ctx.h
	-${CXX} ${CXXFLAGS} -c $(SRC_PATH)/scatter_ctx/scatter_ctx.cpp
	mv scatter_ctx.o $(OBJ_PATH)/scatter_ctx.o

ts_rk.o: $(SRC_PATH)/time_stepping/ts_rk.cpp $(INCLUDE_PATH)/time_stepping/ts_rk.h
	-${CXX} ${CXXFLAGS} -c $(SRC_PATH)/time_stepping/ts_rk.cpp
	mv ts_rk.o $(OBJ_PATH)/ts_rk.o


#.PHONY : clean
init:
	mkdir -p $(BIN_PATH)
	mkdir -p $(OBJ_PATH)

clean::
	-rm -f $(BIN_PATH)/*
	-rm -f $(OBJ_PATH)/*