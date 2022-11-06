export BLISLAB_DIR=..

ifeq ($(BLISLAB_USE_INTEL), true)
include $(BLISLAB_DIR)/make.inc.files/make.intel.inc
else 
ifeq ($(BLISLAB_USE_CLANG),true)
include $(BLISLAB_DIR)/make.inc.files/make.clang.inc
else
include $(BLISLAB_DIR)/make.inc.files/make.gnu.inc
endif
endif

BLISLAB_TEST_CC_SRC= \
									 test_bl_dgemm.c \

BLISLAB_TEST_CPP_SRC=\


OTHER_DEP = \
			                $(LIBBLISLAB) \

#$(info ${OTHER_DEP})

FRAME_CC_SRC=  \
							 ../dgemm/my_dgemm.c \
							 ../dgemm/bl_dgemm_ref.c \
							 ../dgemm/bl_dgemm_util.c \

FRAME_CPP_SRC= \

KERNEL_SRC=    \
							 ../kernels/bl_dgemm_ukr.c \
							 ../kernels/bl_dgemm_rvv_4x4.c \

BLISLAB_OBJ=$(FRAME_CC_SRC:.c=.o) $(FRAME_CPP_SRC:.cpp=.o) $(KERNEL_SRC:.c=.o) $(FRAME_CC_SRC_S:.c=.os) $(KERNEL_SRC_S:.c=.os)


BLISLAB_TEST_EXE= $(BLISLAB_TEST_CC_SRC:.c=.x) $(BLISLAB_TEST_CPP_SRC:.cpp=.x)

all: $(BLISLAB_TEST_EXE)

clean:
	rm -f $(BLISLAB_TEST_EXE)

# ---------------------------------------------------------------------------
# Executable files compiling rules
# ---------------------------------------------------------------------------
%.o: %.c ${OTHER_DEP} 
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)
%.x: %.o ${OTHER_DEP} ${BLISLAB_TEST_CC_SRC:.c=.o}
	$(LD) -v -o $@ ${BLISLAB_TEST_CC_SRC:.c=.o} $(BLISLAB_OBJ) $(LDLIBS) -L${BLISLAB_DIR}/../../../riscv64_github/riscv64-unknown-elf/lib -L${BLISLAB_DIR}/../../../riscv64_github/sysroot/lib
# $(LDLIBS) -L${BLISLAB_DIR}/../../../riscv64_github/sysroot/lib

%.x: %.cpp ${OTHER_DEP}
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)
# ---------------------------------------------------------------------------
