TOPDIR = ../../../../../../InferenceKit/jzdl/4.7.2/
#TOPDIR = ../../../../../../InferenceKit/jzdl/7.2.0/

CROSS_COMPILE:=mips-linux-gnu-

CC = $(CROSS_COMPILE)gcc
CPP = $(CROSS_COMPILE)g++
STRIP = $(CROSS_COMPILE)strip
LIBTYPE ?= muclibc

CXXFLAGS := -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -EL -O3 -march=mips32r2 -mmxu2 -flax-vector-conversions -lpthread -lrt -ldl
#CXXFLAGS := -Wall -EL -O3 -march=mips32r2 -flax-vector-conversions -lpthread -lrt -ldl

ifeq ($(LIBTYPE), muclibc)
	CXXFLAGS += -muclibc
endif

INCLUDES := -I$(TOPDIR)/include 

ifeq ($(LIBTYPE), muclibc)
LIBS := -L$(TOPDIR)/lib/uclibc -ljzdl.m
else
LIBS := -L$(TOPDIR)/lib/glibc -ljzdl.m
endif

ifeq ($(LIBTYPE), muclibc)
	TARGET = facedet_bin_uclibc
else
	TARGET = facedet_bin_glibc
endif

OBJS := inference.o


%.o:%.cpp
	$(CPP) $(INCLUDES) $(CXXFLAGS) -o $@ -c $^

$(TARGET):$(OBJS)
	$(CPP) $(CXXFLAGS) $(OBJS) -o $@ $(INCLUDES) $(LIBS)

all:$(TARGET)

.PHONY: clean
clean: 
	rm -f $(TARGET) $(OBJS)
