CC = g++ -m64 -std=c++17
RM = cmd \/C del
CFLAGS = -Wall -Wextra -fexceptions
DFLAGS = -g
RFLAGS = -O2

BIN_DIR = bin
SRC_DIR = src

OCl_INC = -I Khronos
OCL_LIB = C:/Windows/System32/OpenCL.dll

EXEC_RELEASE = $(BIN_DIR)\proth20.exe
EXEC_DEBUG = $(BIN_DIR)\proth20d.exe

OBJS = src\main.o

build: $(EXEC_RELEASE)

buildd: $(EXEC_DEBUG)

run: $(EXEC_RELEASE)
	$(EXEC_RELEASE)

rund: $(EXEC_DEBUG)
	$(EXEC_DEBUG)

clean:
	$(RM) $(EXEC_RELEASE) $(EXEC_DEBUG)

setrflags:
	$(eval RDFLAGS = $(RFLAGS))
RDFLAGS = $(RFLAGS)

setdflags:
	$(eval RDFLAGS = $(DFLAGS))

$(EXEC_RELEASE): clean setrflags $(OBJS)
	$(CC) $(OBJS) $(RFLAGS) $(OCL_LIB) -static -o $@
	$(RM) $(OBJS)

$(EXEC_DEBUG): clean setdflags $(OBJS)
	$(CC) $(OBJS) $(DFLAGS) $(OCL_LIB) -o $@
	$(RM) $(OBJS)

.cpp.o:
	$(CC) $(CFLAGS) $(RDFLAGS) $(OCl_INC) -c $< -o $@

