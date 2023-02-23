TOPDIR ?= ../../../../InferenceKit/T40/venus/7.2.0/
CROSS_COMPILE:=mips-linux-gnu-

CC = $(CROSS_COMPILE)gcc
CPP = $(CROSS_COMPILE)g++
STRIP = $(CROSS_COMPILE)strip
libtype ?= muclibc
build_type ?= release
#build_type ?= profile
#build_type ?= debug
#build_type ?= nmem

opencv_path=../../../../ThirdParty/opencv_mips_7.2.0_2.26/
CXXFLAGS := -std=c++11 -mfp64 -mnan=2008 -mabs=2008 -Wall -EL -O3 -march=mips32r2 -flax-vector-conversions -lpthread -lrt -ldl -lm

ifeq ($(libtype), muclibc)
	CXXFLAGS += -muclibc
	INCLUDES := -I$(TOPDIR)/include -I$(opencv_path)/uclibc/include/
else
	INCLUDES := -I$(TOPDIR)/include -I$(opencv_path)/glibc/include/
endif

ifeq ($(build_type), release)
    ifeq ($(libtype), muclibc)
        LIBS := -L$(TOPDIR)/lib/uclibc -lvenus -L$(opencv_path)/uclibc/lib/  -lopencv_core -lopencv_highgui -lopencv_imgproc  -lopencv_imgcodecs
    else
        LIBS := -L$(TOPDIR)/lib/glibc -lvenus -L$(opencv_path)/glibc/lib/ -lopencv_core -lopencv_highgui -lopencv_imgproc  -lopencv_imgcodecs
    endif

    ifeq ($(libtype), muclibc)
        TARGET = venus_mnist_bin_uclibc_release
    else
        TARGET = venus_mnist_bin_glibc_release
    endif

else ifeq ($(build_type), profile)
    CXXFLAGS += -DVENUS_PROFILE
    ifeq ($(libtype), muclibc)
        LIBS := -L$(TOPDIR)/lib/uclibc/ -lvenus.p -L$(opencv_path)/uclibc/lib/  -lopencv_core -lopencv_highgui -lopencv_imgproc  -lopencv_imgcodecs
    else
        LIBS := -L$(TOPDIR)/lib/glibc/ -lvenus.p -L$(opencv_path)/uclibc/lib/  -lopencv_core -lopencv_highgui -lopencv_imgproc  -lopencv_imgcodecs
    endif

    ifeq ($(libtype), muclibc)
        TARGET = venus_mnist_bin_uclibc_profile
    else
        TARGET = venus_mnist_bin_glibc_prolfile
    endif

else ifeq ($(build_type), debug)
    CXXFLAGS += -DVENUS_DEBUG
    ifeq ($(libtype), muclibc)
        LIBS := -L$(TOPDIR)/lib/uclibc/ -lvenus.d
    else
        LIBS := -L$(TOPDIR)/lib/glibc/ -lvenus.d
    endif

    ifeq ($(libtype), muclibc)
        TARGET = venus_mnist_bin_uclibc_debug
    else
        TARGET = venus_mnist_bin_glibc_debug
    endif

else ifeq ($(build_type), nmem)
    ifeq ($(libtype), muclibc)
        LIBS := -L$(TOPDIR)/lib/uclibc/ -lvenus.m -L$(opencv_path)/uclibc/lib/  -lopencv_core -lopencv_highgui -lopencv_imgproc  -lopencv_imgcodecs
    else
        LIBS := -L$(TOPDIR)/lib/glibc/ -lvenus.m -L$(opencv_path)/uclibc/lib/  -lopencv_core -lopencv_highgui -lopencv_imgproc  -lopencv_imgcodecs
    endif

    ifeq ($(libtype), muclibc)
        TARGET = venus_mnist_bin_uclibc_nmem
    else
        TARGET = venus_mnist_bin_glibc_nmem
    endif

endif

OBJS := inference.o


%.o:%.cc
	$(CPP) $(INCLUDES) $(CXXFLAGS) -o $@ -c $^

$(TARGET):$(OBJS)
	$(CPP) $(CXXFLAGS) $(OBJS) -o $@ $(INCLUDES) $(LIBS)

all:$(TARGET)

.PHONY: clean
clean: 
	rm -f $(TARGET) $(OBJS)
