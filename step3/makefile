ifeq ($(BLISLAB_USE_INTEL), true)
include $(BLISLAB_DIR)/make.inc.files/make.intel.inc
else 
ifeq ($(BLISLAB_USE_CLANG),true)
include $(BLISLAB_DIR)/make.inc.files/make.clang.inc
else
include $(BLISLAB_DIR)/make.inc.files/make.gnu.inc
endif
endif

$(info * Using CFLAGS=${CFLAGS})
$(info * Using LDFLAGS=${LDFLAGS})

FRAME_CC_SRC=  \
							 dgemm/my_dgemm.c \
							 dgemm/bl_dgemm_ref.c \
							 dgemm/bl_dgemm_util.c \

FRAME_CPP_SRC= \

KERNEL_SRC=    \
							 kernels/bl_dgemm_ukr.c \
							 kernels/bl_dgemm_rvv_4x4.c \
							 kernels/bl_dgemm_asm_rvv_4x4.c \

OTHER_DEP = \
			                 include/bl_dgemm.h \
			                 include/bl_dgemm_kernel.h \

BLISLAB_OBJ=$(FRAME_CC_SRC:.c=.o) $(FRAME_CPP_SRC:.cpp=.o) $(KERNEL_SRC:.c=.o) $(FRAME_CC_SRC_S:.c=.os) $(KERNEL_SRC_S:.c=.os)

all:$(BLISLAB_OBJ) TESTBLISLAB

TESTBLISLAB: $(LIBBLISLAB)
	cd $(BLISLAB_DIR)/test && $(MAKE) && cd $(BLISLAB_DIR)

$(LIBBLISLAB): $(BLISLAB_OBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(BLISLAB_OBJ)
	$(RANLIB) $@

# $(SHAREDLIBBLISLAB): $(BLISLAB_OBJ)
# 	$(LD) -o $@ $(BLISLAB_OBJ) $(LDLIBS) -e=bl_dgemm -L${BLISLAB_DIR}/../../../riscv64_github/sysroot/lib

# ---------------------------------------------------------------------------
# Object files compiling rules
# ---------------------------------------------------------------------------
%.o: %.c $(OTHER_DEP)
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@ $(LDFLAGS)
# ---------------------------------------------------------------------------

clean:
	-rm $(BLISLAB_OBJ) $(LIBBLISLAB) $(SHAREDLIBBLISLAB) dgemm/*~ kernels/*~ kernels/*.o test/*~ include/*~ *~ make.inc.files/*~
	$(MAKE) clean -C test

