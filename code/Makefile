SRC_PATH = src
DEMO_PATH = demo
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
COPTFLAGS		= -O3 -march=native -mtune=native -ffast-math -Rpass-missed=inline -mllvm -inline-threshold=1000
CPPFLAGS		= -DVERSION=${PETSC_VERSION_NUM}

ifeq ($(strip $(order)),)
order		= 4
ORDER_MSG	= order not set. Compiling with $(order)
else
ORDER_MSG	= Compiling with order $(order)
endif
CXX 			= mpicc 
CXXFLAGS		= -std=c++17 $(IFLAGS)

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

LDFLAGS 		= ${PETSC_SYS_LIB}
DETECTED_OS		= $(shell uname -s)
ifneq ($(strip $(DETECTED_OS)),Darwin)
    LDFLAGS += -lstdc++fs
endif

opt: CXXFLAGS += -DNDEBUG $(COPTFLAGS)
opt: $(app)
debug: CXXFLAGS += -DDEBUG $(DEBUGFLAGS)
debug: $(app)	
opt-debug: CXXFLAGS += -DDEBUG $(DEBUGFLAGS) $(COPTFLAGS)
opt-debug: $(app)
all: wave adv_2D adv_1D reflection

# Link object files to create binaries in BIN_PATH/
wave: wave.o io_util.o ts_rk.o scatter_ctx.o create_layout.o
	-${CXX} -o $(BIN_PATH)/$@ $(OBJ_PATH)/wave.o $(OBJ_PATH)/io_util.o  $(OBJ_PATH)/scatter_ctx.o $(OBJ_PATH)/create_layout.o $(OBJ_PATH)/ts_rk.o $(LDFLAGS)

wave_hom: wave_hom.o io_util.o ts_rk.o scatter_ctx.o create_layout.o
	-${CXX} -o $(BIN_PATH)/$@ $(OBJ_PATH)/wave_hom.o $(OBJ_PATH)/io_util.o  $(OBJ_PATH)/scatter_ctx.o $(OBJ_PATH)/create_layout.o $(OBJ_PATH)/ts_rk.o $(LDFLAGS)

adv_2D: adv_2D.o io_util.o ts_rk.o scatter_ctx.o create_layout.o
	-${CXX} -o $(BIN_PATH)/$@ $(OBJ_PATH)/adv_2D.o $(OBJ_PATH)/io_util.o  $(OBJ_PATH)/scatter_ctx.o $(OBJ_PATH)/create_layout.o $(OBJ_PATH)/ts_rk.o $(LDFLAGS)

adv_1D: adv_1D.o io_util.o ts_rk.o scatter_ctx.o create_layout.o
	-${CXX} -o $(BIN_PATH)/$@ $(OBJ_PATH)/adv_1D.o $(OBJ_PATH)/io_util.o  $(OBJ_PATH)/scatter_ctx.o $(OBJ_PATH)/create_layout.o $(OBJ_PATH)/ts_rk.o $(LDFLAGS)

reflection: reflection.o io_util.o ts_rk.o scatter_ctx.o create_layout.o
	-${CXX} -o $(BIN_PATH)/$@ $(OBJ_PATH)/reflection.o $(OBJ_PATH)/io_util.o  $(OBJ_PATH)/scatter_ctx.o $(OBJ_PATH)/create_layout.o $(OBJ_PATH)/ts_rk.o $(LDFLAGS)

# Compile object files to OBJ_PATH/
wave.o: $(DEMO_PATH)/wave/wave_eq_sim.cpp $(DEMO_PATH)/wave/wave_eq_rhs.h $(INCLUDE_PATH)/$(wildcard partitioned_rhs/*.h) $(INCLUDE_PATH)/$(wildcard sbpops/*.h) $(INCLUDE_PATH)/util/vec_util.h
	echo $(ORDER_MSG)
	-${CXX} ${CXXFLAGS} -o $(OBJ_PATH)/$@ -c $(DEMO_PATH)/wave/wave_eq_sim.cpp -DSBP_OPERATOR_ORDER=$(order)

wave_hom.o: $(DEMO_PATH)/wave_hom/wave_eq_hom_sim.cpp $(DEMO_PATH)/wave_hom/wave_eq_hom_rhs.h $(INCLUDE_PATH)/$(wildcard partitioned_rhs/*.h) $(INCLUDE_PATH)/$(wildcard sbpops/*.h) $(INCLUDE_PATH)/util/vec_util.h
	echo $(ORDER_MSG)
	-${CXX} ${CXXFLAGS} -o $(OBJ_PATH)/$@ -c $(DEMO_PATH)/wave_hom/wave_eq_hom_sim.cpp -DSBP_OPERATOR_ORDER=$(order)

adv_2D.o: $(DEMO_PATH)/advection/advection_2D_sim.cpp $(DEMO_PATH)/advection/advection_rhs.h $(INCLUDE_PATH)/$(wildcard partitioned_rhs/*.h) $(INCLUDE_PATH)/$(wildcard sbpops/*.h) $(INCLUDE_PATH)/util/vec_util.h
	echo $(ORDER_MSG)
	-${CXX} ${CXXFLAGS} -o $(OBJ_PATH)/$@ -c $(DEMO_PATH)/advection/advection_2D_sim.cpp -DSBP_OPERATOR_ORDER=$(order)

adv_1D.o: $(DEMO_PATH)/advection/advection_1D_sim.cpp $(DEMO_PATH)/advection/advection_rhs.h $(INCLUDE_PATH)/$(wildcard partitioned_rhs/*.h) $(INCLUDE_PATH)/$(wildcard sbpops/*.h) $(INCLUDE_PATH)/util/vec_util.h
	echo $(ORDER_MSG)
	-${CXX} ${CXXFLAGS} -o $(OBJ_PATH)/$@ -c $(DEMO_PATH)/advection/advection_1D_sim.cpp -DSBP_OPERATOR_ORDER=$(order)

reflection.o: $(DEMO_PATH)/reflection/reflection_sim.cpp $(DEMO_PATH)/reflection/reflection_rhs.h $(INCLUDE_PATH)/$(wildcard partitioned_rhs/*.h) $(INCLUDE_PATH)/$(wildcard sbpops/*.h) $(INCLUDE_PATH)/util/vec_util.h
	echo $(ORDER_MSG)
	-${CXX} ${CXXFLAGS} -o $(OBJ_PATH)/$@ -c $(DEMO_PATH)/reflection/reflection_sim.cpp -DSBP_OPERATOR_ORDER=$(order)	
	
create_layout.o: $(SRC_PATH)/grids/create_layout.cpp  $(INCLUDE_PATH)/grids/create_layout.h $(INCLUDE_PATH)/grids/layout.h
	-${CXX} ${CXXFLAGS} -o $(OBJ_PATH)/$@ -c $(SRC_PATH)/grids/create_layout.cpp

io_util.o: $(SRC_PATH)/util/io_util.cpp $(INCLUDE_PATH)/util/io_util.h
	-${CXX} ${CXXFLAGS} -o $(OBJ_PATH)/$@ -c $(SRC_PATH)/util/io_util.cpp

scatter_ctx.o: $(SRC_PATH)/scatter_ctx/scatter_ctx.cpp $(INCLUDE_PATH)/scatter_ctx/scatter_ctx.h
	-${CXX} ${CXXFLAGS} -o $(OBJ_PATH)/$@ -c $(SRC_PATH)/scatter_ctx/scatter_ctx.cpp

ts_rk.o: $(SRC_PATH)/time_stepping/ts_rk.cpp $(INCLUDE_PATH)/time_stepping/ts_rk.h
	-${CXX} ${CXXFLAGS} -o $(OBJ_PATH)/$@ -c $(SRC_PATH)/time_stepping/ts_rk.cpp


#.PHONY : clean
init:
	mkdir -p $(BIN_PATH)
	mkdir -p $(OBJ_PATH)

clean::
	-rm -f $(BIN_PATH)/*
	-rm -f $(OBJ_PATH)/*