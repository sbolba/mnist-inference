# ============================================================================
# MNIST Transformer Inference - Makefile Ottimizzato
# Supporta: Windows (MinGW/MSYS2/Git Bash) e POSIX (Linux/macOS)
# ============================================================================

# Compilatore
CC = gcc

# ============================================================================
# Rilevamento Sistema Operativo
# ============================================================================
# Controlla se abbiamo una shell POSIX-like (MSYS2, Git Bash, Cygwin)
SHELL_TYPE := $(shell echo $$0)

ifeq ($(OS),Windows_NT)
    PLATFORM = windows
    EXE_EXT = .exe
    
    # Controlla se stiamo usando una shell POSIX su Windows
    # Verifica se mkdir -p è disponibile (MSYS2, Git Bash, Cygwin)
    HAS_MKDIR_P := $(shell mkdir -p /tmp/__mkdir_test 2>/dev/null && echo 1 || echo 0)
    ifeq ($(HAS_MKDIR_P),1)
        # MSYS2/Git Bash - usa comandi POSIX
        USE_POSIX_SHELL = 1
        MKDIR = mkdir -p
        RM = rm -f
        RMDIR = rm -rf
    else
        # CMD.exe nativa
        USE_POSIX_SHELL = 0
        MKDIR = mkdir
        RM = del /Q
        RMDIR = rmdir /S /Q
    endif
    
    # Flags Windows
    CFLAGS_PLATFORM = 
    LDFLAGS_PLATFORM = 
else
    # Linux/macOS
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        PLATFORM = linux
    else ifeq ($(UNAME_S),Darwin)
        PLATFORM = macos
    else
        PLATFORM = posix
    endif
    
    USE_POSIX_SHELL = 1
    EXE_EXT =
    MKDIR = mkdir -p
    RM = rm -f
    RMDIR = rm -rf
    
    # Flags POSIX (OpenMP + pthread)
    CFLAGS_PLATFORM = -fopenmp
    LDFLAGS_PLATFORM = -lpthread -fopenmp
endif

# ============================================================================
# Configurazione Directory
# ============================================================================
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin
INCLUDE_DIR = include

# Sottodirectory sorgenti
CORE_DIR = $(SRC_DIR)/core
OPS_DIR = $(SRC_DIR)/ops
MODELS_DIR = $(SRC_DIR)/mnist_model
UTILS_DIR = $(SRC_DIR)/utils
TRANSFORMER_DIR = $(SRC_DIR)/visual_transformer
TRANSFORMER_DIR = $(SRC_DIR)/visual_transformer


# ============================================================================
# Flags di Compilazione
# ============================================================================
# Flags comuni
CFLAGS_BASE = -Wall -Wextra -I$(INCLUDE_DIR)

# Ottimizzazioni
CFLAGS_OPT = -O3 -march=native -ffast-math

# Flags di debug
CFLAGS_DEBUG = -g -O0 -DDEBUG

# Flags finali
CFLAGS = $(CFLAGS_BASE) $(CFLAGS_OPT) $(CFLAGS_PLATFORM)

# Linker flags
LDFLAGS = -lm $(LDFLAGS_PLATFORM)

# ============================================================================
# Selezione File Platform-Specific
# ============================================================================
ifeq ($(PLATFORM),windows)
    MMAP_SRC = $(CORE_DIR)/mmap_loader_windows.c
else
    MMAP_SRC = $(CORE_DIR)/mmap_loader_posix.c
endif

# ============================================================================
# Raccolta File Sorgenti
# ============================================================================
# File comuni (escludi file platform-specific)
CORE_COMMON = $(filter-out %_windows.c %_posix.c, $(wildcard $(CORE_DIR)/*.c))
OPS_SRC = $(wildcard $(OPS_DIR)/*.c)
MODELS_SRC = $(wildcard $(MODELS_DIR)/*.c)
UTILS_SRC = $(wildcard $(UTILS_DIR)/*.c)
TRANSFORMER_SRC = $(wildcard $(TRANSFORMER_DIR)/*.c)
MAIN_SRC = $(wildcard $(SRC_DIR)/*.c)

# Tutti i sorgenti
SOURCES = $(CORE_COMMON) $(MMAP_SRC) $(OPS_SRC) $(MODELS_SRC) $(UTILS_SRC) $(TRANSFORMER_SRC) $(MAIN_SRC)

# File oggetto
OBJECTS = $(SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

# Dipendenze header
HEADERS = $(wildcard $(INCLUDE_DIR)/*.h)

# Target finale
TARGET = $(BIN_DIR)/mnist_inference$(EXE_EXT)

# ============================================================================
# Colori per Output (solo su POSIX)
# ============================================================================
ifeq ($(PLATFORM),windows)
    GREEN =
    YELLOW =
    BLUE =
    NC =
    CHECK = [OK]
    INFO = [INFO]
    CLEAN = [CLEAN]
else
    GREEN = \033[0;32m
    YELLOW = \033[0;33m
    BLUE = \033[0;34m
    NC = \033[0m
    CHECK = ✅
    INFO = ℹ️
    CLEAN = 🧹
endif

# ============================================================================
# Regole Principali
# ============================================================================
.PHONY: all clean debug release run test info help dirs

# Default target
all: run

# Build con ottimizzazioni
release: CFLAGS = $(CFLAGS_BASE) $(CFLAGS_OPT) $(CFLAGS_PLATFORM)
release: dirs $(TARGET)
	@echo "$(GREEN)$(CHECK) Compilazione completata: $(TARGET)$(NC)"
	@echo "$(BLUE)$(INFO) Platform: $(PLATFORM)$(NC)"

# Build con debug
debug: CFLAGS = $(CFLAGS_BASE) $(CFLAGS_DEBUG) $(CFLAGS_PLATFORM)
debug: clean dirs $(TARGET)
	@echo "$(GREEN)$(CHECK) Debug build completato$(NC)"

# ============================================================================
# Creazione Directory - Funziona su MSYS2/Git Bash e CMD
# ============================================================================
dirs:
	@$(MKDIR) $(BUILD_DIR)/core
	@$(MKDIR) $(BUILD_DIR)/ops
	@$(MKDIR) $(BUILD_DIR)/mnist_model
	@$(MKDIR) $(BUILD_DIR)/utils
	@$(MKDIR) $(BUILD_DIR)/visual_transformer
	@$(MKDIR) $(BIN_DIR)

# ============================================================================
# Compilazione
# ============================================================================
# Linking
$(TARGET): $(OBJECTS)
	@echo "$(BLUE)$(INFO) Linking $(notdir $@)...$(NC)"
	@$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

# Compilazione file oggetto - pattern per ogni sottodirectory
$(BUILD_DIR)/core/%.o: $(CORE_DIR)/%.c $(HEADERS)
	@echo "$(BLUE)$(INFO) Compiling $(notdir $<)...$(NC)"
	@$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/ops/%.o: $(OPS_DIR)/%.c $(HEADERS)
	@echo "$(BLUE)$(INFO) Compiling $(notdir $<)...$(NC)"
	@$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/mnist_model/%.o: $(MODELS_DIR)/%.c $(HEADERS)
	@echo "$(BLUE)$(INFO) Compiling $(notdir $<)...$(NC)"
	@$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/utils/%.o: $(UTILS_DIR)/%.c $(HEADERS)
	@echo "$(BLUE)$(INFO) Compiling $(notdir $<)...$(NC)"
	@$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/visual_transformer/%.o: $(TRANSFORMER_DIR)/%.c $(HEADERS)
	@echo "$(BLUE)$(INFO) Compiling $(notdir $<)...$(NC)"
	@$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(HEADERS)
	@echo "$(BLUE)$(INFO) Compiling $(notdir $<)...$(NC)"
	@$(CC) $(CFLAGS) -c $< -o $@

# ============================================================================
# Utility Targets
# ============================================================================
# Esegui il programma
run: release
	@echo "$(BLUE)$(INFO) Esecuzione $(TARGET)...$(NC)"
	@./$(TARGET)

# Test
test: release
	@echo "$(BLUE)$(INFO) Esecuzione tests...$(NC)"
ifeq ($(USE_POSIX_SHELL),1)
	@./tests/run_tests.sh || echo "Test suite non trovata"
else
	@echo "Test suite non implementata"
endif

# Pulizia completa
clean:
	@echo "$(YELLOW)$(CLEAN) Rimozione file compilati...$(NC)"
	@$(RMDIR) $(BUILD_DIR) 2>/dev/null || true
	@$(RMDIR) $(BIN_DIR) 2>/dev/null || true
	@echo "$(YELLOW)$(CLEAN) Pulizia completata$(NC)"

# Pulizia solo oggetti
clean-obj:
	@echo "$(YELLOW)$(CLEAN) Rimozione file oggetto...$(NC)"
	@$(RMDIR) $(BUILD_DIR) 2>/dev/null || true

# ============================================================================
# Build Speciali
# ============================================================================
# Build con profiling
profile: CFLAGS = $(CFLAGS_BASE) -O2 -pg $(CFLAGS_PLATFORM)
profile: LDFLAGS += -pg
profile: clean dirs $(TARGET)
	@echo "$(GREEN)$(CHECK) Profile build completato$(NC)"
	@echo "$(BLUE)$(INFO) Esegui il programma e poi: gprof $(TARGET) gmon.out$(NC)"

# Build con ottimizzazioni aggressive
fast: CFLAGS = $(CFLAGS_BASE) -O3 -march=native -mtune=native -ffast-math -funroll-loops -flto $(CFLAGS_PLATFORM)
fast: LDFLAGS += -flto
fast: clean dirs $(TARGET)
	@echo "$(GREEN)$(CHECK) Fast build completato con LTO$(NC)"

# ============================================================================
# Informazioni
# ============================================================================
# Mostra configurazione
info:
	@echo "=========================================="
	@echo "Build Configuration"
	@echo "=========================================="
	@echo "Platform:     $(PLATFORM)"
	@echo "Shell:        $(if $(USE_POSIX_SHELL),POSIX,CMD)"
	@echo "Compiler:     $(CC)"
	@echo "CFLAGS:       $(CFLAGS)"
	@echo "LDFLAGS:      $(LDFLAGS)"
	@echo "Target:       $(TARGET)"
	@echo "=========================================="
	@echo "Source Files:"
	@echo "  Core:       $(words $(CORE_COMMON)) common + 1 platform"
	@echo "  Ops:        $(words $(OPS_SRC))"
	@echo "  Mnist_Model:     $(words $(MODELS_SRC))"
	@echo "  Utils:      $(words $(UTILS_SRC))"
	@echo "  Main:       $(words $(MAIN_SRC))"
	@echo "  Total:      $(words $(SOURCES))"
	@echo "=========================================="

# Help
help:
	@echo "MNIST Transformer Inference - Makefile"
	@echo ""
	@echo "Targets disponibili:"
	@echo "  make / make all      - Build release + run (default)"
	@echo "  make release         - Build con ottimizzazioni"
	@echo "  make debug           - Build con debug symbols"
	@echo "  make fast            - Build con ottimizzazioni aggressive + LTO"
	@echo "  make profile         - Build per profiling (gprof)"
	@echo "  make run             - Compila ed esegui"
	@echo "  make test            - Esegui test suite"
	@echo "  make clean           - Rimuovi tutti i file compilati"
	@echo "  make clean-obj       - Rimuovi solo file oggetto"
	@echo "  make info            - Mostra configurazione build"
	@echo "  make help            - Mostra questo messaggio"
	@echo ""
	@echo "Platform: $(PLATFORM)"
	@echo "Shell:    $(if $(USE_POSIX_SHELL),POSIX-compatible,Windows CMD)"