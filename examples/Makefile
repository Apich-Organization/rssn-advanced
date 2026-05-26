CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -I..
LIB_DIR = ../target/release
LIB     = rssn_advanced

# Windows (MinGW/MSYS2): no rpath — the DLL is found via PATH or the
# directory that the binary lives in.  Linux: embed rpath so the binary
# locates the .so at runtime without LD_LIBRARY_PATH.
ifeq ($(OS),Windows_NT)
    LDFLAGS = -L$(LIB_DIR) -l$(LIB)
else
    LDFLAGS = -L$(LIB_DIR) -l$(LIB) -Wl,-rpath,$(LIB_DIR)
endif

TARGETS = basics

.PHONY: all run clean

all: $(TARGETS)

basics: basics.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# On Windows copy the DLL next to the binary before running.
run: basics
ifeq ($(OS),Windows_NT)
	cp $(LIB_DIR)/$(LIB).dll . 2>/dev/null || true
endif
	./basics

clean:
	rm -f $(TARGETS) $(LIB).dll
