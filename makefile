all: myPaint test run

myPaint: main.cu
	nvcc -o myPaint main.cu

test: test.c
	gcc -o test test.c

run:
	./test

clean:
	rm -r myPaint test speedup.txt test_output
