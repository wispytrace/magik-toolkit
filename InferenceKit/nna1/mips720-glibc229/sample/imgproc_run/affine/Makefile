TOPDIR = ../../../2.26
CROSS_COMPILE:=mips-linux-gnu-

CC = $(CROSS_COMPILE)gcc
CPP = $(CROSS_COMPILE)g++
STRIP = $(CROSS_COMPILE)strip
LIBTYPE ?= muclibc
BUILD_TYPE ?= release
# BUILD_TYPE ?= profile
# BUILD_TYPE ?= debug


CXXFLAGS := -std=c++11 -mfp64 -mnan=2008 -mabs=2008 -Wall -EL -O3 -march=mips32r2 -flax-vector-conversions -lpthread -lrt -ldl -lm

ifeq ($(LIBTYPE), muclibc)
	CXXFLAGS += -muclibc
endif

INCLUDES := -I$(TOPDIR)/include

ifeq ($(BUILD_TYPE), release)
    ifeq ($(LIBTYPE), muclibc)
	LIBS := -L$(TOPDIR)/lib/uclibc -lvenus
    else
	LIBS := -L$(TOPDIR)/lib/glibc -lvenus
    endif
else ifeq ($(BUILD_TYPE), profile)
    CXXFLAGS += -DVENUS_PROFILE
    ifeq ($(LIBTYPE), muclibc)
	LIBS := -L$(TOPDIR)/lib/uclibc/ -lvenus.p
    else
	LIBS := -L$(TOPDIR)/lib/glibc/ -lvenus.p
    endif
else
    ifeq ($(LIBTYPE), muclibc)
	LIBS := -L$(TOPDIR)/lib/uclibc -lvenus.d
    else
	LIBS := -L$(TOPDIR)/lib/glibc -lvenus.d
    endif
endif

ifeq ($(LIBTYPE), muclibc)
	TARGET = affine_bin_uclibc_public
else
	TARGET = affine_bin_glibc_public
endif

OBJS := affine_run.o


%.o:%.cc
	$(CPP) $(INCLUDES) $(CXXFLAGS) -o $@ -c $^

$(TARGET):$(OBJS)
	$(CPP) $(CXXFLAGS) $(OBJS) -o $@ $(INCLUDES) $(LIBS)

all:$(TARGET)

.PHONY: clean
clean:
	rm -f $(TARGET) $(OBJS)
