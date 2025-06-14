# Compiler and flags
CC = g++
CFLAGS = -std=c++17 -I/usr/include/jsoncpp $(shell python3 -m pybind11 --includes) -fPIC -Wall
LDFLAGS = -shared -ldl -L/usr/lib/x86_64-linux-gnu/ -L/usr/local/lib -L/usr/lib -pthread -ljsoncpp  $(shell python3-config --ldflags) -lpython3.11

TARGET = libmessage.so
SOURCES = libmessageunix.cpp packetanalyzer.cpp evaluate.cpp
OBJECTS = $(SOURCES:.cpp=.o)

# Default target to build the shared library
all: $(TARGET)

# Rule to compile each .cpp file into a .o file
%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to build the shared library from object files
$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $(OBJECTS) $(LDFLAGS)

# Clean rule to remove generated files
clean:
	rm -f $(TARGET) $(OBJECTS)
