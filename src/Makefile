CC = gcc
CFLAGS = -O0 -g -I.
LDFLAGS = -lm

TARGET = dr_test
SRCS = main.c engine.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# ---- tests ----
TEST_DEPS = engine.c engine.h tests/test_util.h

tests/test_engine: tests/test_engine.c $(TEST_DEPS)
	$(CC) $(CFLAGS) tests/test_engine.c engine.c -o $@ $(LDFLAGS)

tests/test_render: tests/test_render.c $(TEST_DEPS) vec3.h camera.h renderer.h image.h
	$(CC) $(CFLAGS) tests/test_render.c engine.c -o $@ $(LDFLAGS)

tests/test_gradcheck: tests/test_gradcheck.c $(TEST_DEPS) vec3.h camera.h renderer.h
	$(CC) $(CFLAGS) tests/test_gradcheck.c engine.c -o $@ $(LDFLAGS)

test-engine: tests/test_engine
	./tests/test_engine

test-render: tests/test_render
	./tests/test_render

test-gradcheck: tests/test_gradcheck
	./tests/test_gradcheck

test: test-engine test-render test-gradcheck

clean:
	rm -f $(OBJS) $(TARGET) tests/test_engine tests/test_render tests/test_gradcheck

.PHONY: all clean test test-engine test-render test-gradcheck
