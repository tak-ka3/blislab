CC  = gcc
CXX = g++

ARCH = ar
ARCHFLAGS = cr
RANLIB = ranlib

ifeq ($(BLISLAB_USE_BLAS),true)
CFLAGS = -$(COMPILER_OPT_LEVEL) -march=core-avx2 -fPIC -DUSE_BLAS
#LDLIBS = $(LIBBLISLAB) -lpthread -lm -lblas                                 #sudo apt-get install libblas-dev (Netlib BLAS)
#LDLIBS = $(LIBBLISLAB) ${BLAS_DIR}/lib/libblis.a -lpthread -lm              #blis: https://github.com/flame/blis/
#LDLIBS = $(LIBBLISLAB) ${BLAS_DIR}/lib/libopenblas.a -lpthread -lm          #openblas: https://github.com/xianyi/OpenBLAS
LDLIBS = $(LIBBLISLAB) ${BLAS_DIR}/lib/*.a -lpthread -lm                     #*/General for any static link file: .a
else
CFLAGS = -$(COMPILER_OPT_LEVEL) -march=native -fPIC
LDLIBS = $(LIBBLISLAB) -lpthread -lm
endif

LIBBLISLAB = $(BLISLAB_DIR)/lib/libblislab.a
SHAREDLIBBLISLAB = $(BLISLAB_DIR)/lib/libblislab.so

LDFLAGS = -I$(BLISLAB_DIR)/include -I$(BLISLAB_DIR)/kernels -I${BLAS_DIR}/include

OS_NAME = $(shell uname -s)
ifeq ($(OS_NAME),Linux)
LDLIBS += -lrt
endif



CC  = riscv64-unknown-elf-gcc
CXX = ../../build/bin/clang++
LD = riscv64-unknown-elf-gcc

ARCH = ar
ARCHFLAGS = cr
RANLIB = ranlib

ifeq ($(BLISLAB_USE_BLAS),true)
CFLAGS = -$(COMPILER_OPT_LEVEL) -v -march=core-avx2 -fPIC -DUSE_BLAS
#LDLIBS = $(LIBBLISLAB) -lpthread -lm -lblas                                 #sudo apt-get install libblas-dev (Netlib BLAS)
#LDLIBS = $(LIBBLISLAB) ${BLAS_DIR}/lib/libblis.a -lpthread -lm              #blis: https://github.com/flame/blis/
#LDLIBS = $(LIBBLISLAB) ${BLAS_DIR}/lib/libopenblas.a -lpthread -lm          #openblas: https://github.com/xianyi/OpenBLAS
LDLIBS = $(LIBBLISLAB) ${BLAS_DIR}/lib/*.a -lpthread -lm                     #*/General for any static link file: .a
else
CFLAGS = -$(COMPILER_OPT_LEVEL) -v -march=rv64gcv
LDLIBS = $(LIBBLISLAB) -lpthread -lm
endif

LIBBLISLAB = $(BLISLAB_DIR)/lib/libblislab.a
SHAREDLIBBLISLAB = $(BLISLAB_DIR)/lib/libblislab.so

LDFLAGS = -I$(BLISLAB_DIR)/include -I$(BLISLAB_DIR)/kernels -I${BLAS_DIR}/include -I${BLISLAB_DIR}/../../../riscv64_github/riscv64-unknown-elf/include -I${BLISLAB_DIR}/../../clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04/lib/clang/14.0.0/include

OS_NAME = $(shell uname -s)
ifeq ($(OS_NAME),Linux)
LDLIBS += -lrt
endif

