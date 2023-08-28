# CXX=/home/s1576972/local/bin/g++
CPPFLAGS=-std=c++17 -fpic -MMD -MP -Wall -Wextra -Winit-self \
		 -Wno-missing-field-initializers
LDFLAGS=-Wall --shared
INCLUDE=-I../include -I/data/s1576972/archive/eigen-3.3.9/

program_dir=../lib/dev
program_name=libuniquelist.so
program=$(program_dir)/$(program_name)

source=main.cpp
object=$(source:.cpp=.o)
dependency=$(object:.o=.d)

.PHONY: all clean help

all: $(program)   ## compile all

$(program): $(object)
	mkdir -p $(program_dir)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(INCLUDE) -o $@ -c $<

clean:   ## clean up object directory and program directory
	-rm -rf $(program_dir) $(object)

-include $(dependency)
