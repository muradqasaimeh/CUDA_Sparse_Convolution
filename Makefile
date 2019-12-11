# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda
CUDNN_PATH := usr/local/cuda-8.0/cudnn
##########################################################

## CC COMPILER OPTIONS ##

# CC compiler options:
CC=g++
CC_FLAGS=-I$(CURDIR)/include -I$(CUDA_ROOT_DIR)/include -I$(CUDNN_PATH)/include
CC_LIBS=-L$(CUDA_ROOT_DIR)/lib64 -L$(CUDNN_PATH)/lib64 -L/usr/local/lib -lcublas 

##########################################################

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS= -arch=sm_35 -std=c++11 -O2 -I$(CURDIR)/include 
NVCC_LIBS=

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64 -L$(CUDNN_PATH)/lib64 -L/usr/local/lib
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include -I$(CUDNN_PATH)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lpthread -lcudart -lcublas -lcudnn -lcusparse
 

##########################################################

## Project file structure ##

# Source file directory:
SRC_DIR = $(CURDIR)/src

# Object file directory:
OBJ_DIR = $(CURDIR)/bin

# Include header file diretory:
INC_DIR = $(CURDIR)/include

##########################################################

## Make variables ##

# Target executable name:
EXE = main

# Object files:
OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/CONV_cuDNN.o $(OBJ_DIR)/utils.o $(OBJ_DIR)/im2col.o $(OBJ_DIR)/col2im.o $(OBJ_DIR)/CONV_ref.o $(OBJ_DIR)/CONV_cuBLAS.o $(OBJ_DIR)/CONV_cuSPARSE.o $(OBJ_DIR)/CONV_CUDA.o $(OBJ_DIR)/data_reshape.o $(OBJ_DIR)/CONV_proposed.o

##########################################################

## Compile ##

# Link c++ and CUDA compiled object files to target executable:
$(EXE) : $(OBJS)
	$(CC) $(CC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# Compile main .cpp file to object files:
$(OBJ_DIR)/%.o : %.cpp
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile C++ source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp $(INC_DIR)/%.h
	$(CC) $(CC_FLAGS) -c $< -o $@  

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Clean objects in object directory.
clean:
	$(RM) bin/* *.o $(EXE)






