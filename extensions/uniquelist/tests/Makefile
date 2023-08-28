CFLAGS                    = -std=c++20 -MMD -MP \
                              -Wall -Wextra -Winit-self \
                              -Wno-missing-field-initializers
LDFLAGS                   = -Wall \
                              -L /data/s1576972/local/lib64/ \
                              -L /data/s1576972/local/lib/ \
                              -L $(HOME)/local/lib \
                              -lmmio -lgtest -lgtest_main -pthread
INCLUDE                   = -I ../include \
                              -I $(HOME)/local/include \
                              -I /data/s1576972/local/include/

ifeq "$(strip $(NSMAKE_COMPILER))" ""
	COMPILER = g++
else
	COMPILER = $(NSMAKE_COMPILER)
endif

# Adjust the width of the first column of help.
help_column_widht         = 20
program_basename          = test.out
# Source file names.
source                    = $(shell find . -regex .*\.cpp | cut -c 3-)
# source                    = test_v1_utils_unique_list.cpp
# File names of executables.
obj_dir                   = obj
object                    = $(addprefix $(obj_dir)/, $(source:.cpp=.o))
dependency                = $(addprefix $(obj_dir)/, $(source:.cpp=.d))
program                   = $(addprefix $(obj_dir)/, $(program_basename))

.PHONY: clean help test

test: $(program)
	LD_LIBRARY_PATH=$(HOME)/local/lib:/data/s1576972/local/lib64 ./$(program)

$(program): $(object)
	$(COMPILER) -o $@ $^ $(LDFLAGS)

$(obj_dir)/%.a: $(obj_dir)/%.o
	$(COMPILER) -o $@ $^ $(LDFLAGS)

$(obj_dir)/%.o: %.cpp
	-mkdir -p $(obj_dir)
	$(COMPILER) $(CFLAGS) $(INCLUDE) -o $@ -c $<

clean:   ## clean up object directory and program directory
	-rm -rf *.o *.d *.a obj

help:   ## show this message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile | sort | \
		awk 'BEGIN {FS = ":.*?## "}; \
		{printf "\033[36m%-$(help_column_widht)s\033[0m %s\n", $$1, $$2}'

-include $(dependency)
