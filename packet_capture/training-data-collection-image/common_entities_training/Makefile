CC = g++
CFLAGS = -std=c++17 -fPIC -Wall
LDFLAGS = -shared -ldl 

TARGET = libmessage_training.so
SOURCES = libmessageunix_training.cpp 
OBJECTS = $(SOURCES:.cpp=.o)

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
