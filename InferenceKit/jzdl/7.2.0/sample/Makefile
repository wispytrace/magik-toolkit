TOPDIR = ..
CROSS_COMPILE:=mips-linux-gnu-

CC = $(CROSS_COMPILE)gcc
CPP = $(CROSS_COMPILE)g++
STRIP = $(CROSS_COMPILE)strip
LIBTYPE ?= muclibc

CXXFLAGS := -Wall -EL -O3 -march=mips32r2 -flax-vector-conversions -lpthread -lrt -ldl

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
	TARGET = mnist_bin_uclibc
else
	TARGET = mnist_bin_glibc
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
