#compiler
OMPL_DIR = /usr
INC_CLASSES = ./proj_classes/
INC_PLANNERS = ./planners/
INC_VALIDITY = ./validity_checkers/
INC_RUN = ./run/

EIGEN_DIR = /home/avishai/Documents/eigen
KDL_DIR = /usr/local

CXX= g++
CXXFLAGS= -I${OMPL_DIR}/local/include -I${OMPL_DIR}/lib/x86_64-linux-gnu -I${INC_CLASSES} -I${INC_PLANNERS} -I${KDL_DIR}/include -I$(EIGEN_DIR) 
LDFLAGS=  -L${OMPL_DIR}/local/lib -L${OMPL_DIR}/lib/x86_64-linux-gnu -lompl -lboost_filesystem -lboost_system -lboost_serialization -lboost_program_options -Wl,-rpath ${OMPL_DIR}/lib/x86_64-linux-gnu -lorocos-kdl 
LIBS += -L/usr/lib/x86_64-linux-gnu -lboost_system

CPPGDKDL = ${INC_CLASSES}kdl_class.cpp 
CPPGD = ${INC_CLASSES}kdl_class.cpp ${INC_VALIDITY}StateValidityCheckerGD.cpp

CPP_P = ${INC_RUN}plan.cpp ${INC_PLANNERS}CBiRRT_gd.cpp ${INC_PLANNERS}SBL_gd.cpp ${INC_PLANNERS}RRT_gd.cpp


all:

	$(CXX) ${CPP_P} ${CPPGD} ${CPPVRF} -o pgd $(CXXFLAGS) $(LDFLAGS) -DPGD -std=c++11


