TOPDIR = ./
CROSS_COMPILE:=

CC = $(CROSS_COMPILE)gcc
CPP = $(CROSS_COMPILE)g++
STRIP = $(CROSS_COMPILE)strip

CXXFLAGS := -Wall -g -std=c++11 -lrt -pthread -mavx -ldl

INCLUDES := -I$(TOPDIR)/include

LIBS := -L$(TOPDIR)/lib -lmagikExecutor -lprotobuf -lprotobuf-lite -Wl,-rpath ./lib

TARGET = pc_inference_bin

OBJS :=inference.o

%.o:%.cc
	$(CPP) $(INCLUDES) $(CXXFLAGS) -o $@ -c $^

$(TARGET):$(OBJS)
	$(CPP) $(CXXFLAGS) $(OBJS) -o $@ $(INCLUDES) $(LIBS)

all:$(TARGET)

.PHONY: clean
clean: 
	rm -f $(TARGET) $(OBJS)
