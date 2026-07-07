# Makefile for the chess engine.
#
# Build with WinLibs GCC 16 (64-bit). Use the accompanying build.ps1 on Windows,
# which puts the compiler on PATH and invokes mingw32-make, or run
#   mingw32-make            (release build -> bin/engine.exe)
#   mingw32-make perft      (build + run perft self-test)
#   mingw32-make clean

CXX      ?= g++

# Binary name: engine.exe on Windows, engine (no extension) on Linux/macOS so the
# Lichess deploy (lichess-bot config, CI perft step) finds bin/engine.
ifeq ($(OS),Windows_NT)
  EXE    ?= bin/engine.exe
else
  EXE    ?= bin/engine
endif

SRCDIR   := src
OBJDIR   := build
SOURCES  := $(wildcard $(SRCDIR)/*.cpp)
OBJECTS  := $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SOURCES))
DEPS     := $(OBJECTS:.o=.d)

# -march=native tunes for the build machine. Override ARCH for a deploy target,
# e.g. ARCH="-mavx2 -mbmi2 -mpopcnt" for broad x86-64-v3 compatibility.
ARCH     ?= -march=native
CXXSTD   := -std=c++20
OPT      := -O3 -funroll-loops -flto=auto
WARN     := -Wall -Wextra -Wno-unused-parameter
DEFS     := -DNDEBUG

# Cross-platform threading/linking. The engine uses std::thread (Lazy SMP):
# Linux requires -pthread, and full -static often fails there, so only fully
# static-link on Windows (standalone .exe); elsewhere static-link just the
# GCC/stdc++ runtimes for a portable-enough binary.
ifeq ($(OS),Windows_NT)
  THREADS :=
  LDFLAGS := -static -static-libgcc -static-libstdc++ -flto=auto
else
  THREADS := -pthread
  LDFLAGS := -pthread -static-libgcc -static-libstdc++ -flto=auto
endif

CXXFLAGS := $(CXXSTD) $(OPT) $(ARCH) $(THREADS) $(WARN) $(DEFS) -MMD -MP -I$(SRCDIR)

.PHONY: all clean perft run

all: $(EXE)

$(EXE): $(OBJECTS) | bin
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)
	@echo Built $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

bin:
	mkdir -p bin

perft: $(EXE)
	$(EXE) perft

run: $(EXE)
	$(EXE)

clean:
	rm -rf $(OBJDIR) bin

-include $(DEPS)
