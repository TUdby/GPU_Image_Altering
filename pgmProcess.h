#ifndef pgmProcess_h
#define pgmProcess_h

void serialDrawCircle(int *pixels, int numCols, int numRows, int centerCol, int centerRow, int radius);
__global__ void cudaDrawCircle(int *pixels, int numCols, int numRows, int centerCol, int centerRow);

void serialDrawEdge(int *pixels, int numRows, int numCols, int edgewidth);
__global__ void cudaDrawEdge(int *pixels, int numRows, int numCols, int edgewidth);

void serialDrawLine(int *pixels, int col1, int row1, int col2, int row2, int numCols, int numRows, int bound);
__global__ void cudaDrawLine(int *pixels, double slope, int col1, int row1, int numCols, int n);

#endif