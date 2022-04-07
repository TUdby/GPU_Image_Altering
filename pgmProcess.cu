#include "pgmProcess.h"

void serialDrawCircle(int *pixels, int numCols, int numRows, int centerCol, int centerRow, int radius) {
   int a_squared, b_squared, c_squared;
   
   for(int row = 0; row < numRows; row++) {
      for(int col = 0; col < numCols; col++) {
         a_squared = (row - centerRow) * (row - centerRow);
         b_squared = (col - centerCol) * (col - centerCol);
         c_squared = a_squared + b_squared;

         if(c_squared <= radius * radius)
            pixels[row * numCols + col] = 0;
      }
   }
}

__global__ void cudaDrawCircle(int *pixels, int numCols, int numRows, int centerCol, int centerRow, int radius) {
   int col = blockIdx.x * blockDim.x + threadIdx.x;
   int row = blockIdx.y * blockDim.y + threadIdx.y;

   int a_squared, b_squared, c_squared;
   if(col < numCols && row < numRows) {
      a_squared = (row - centerRow) * (row - centerRow);
      b_squared = (col - centerCol) * (col - centerCol);
      c_squared = a_squared + b_squared;

      if(c_squared <= radius * radius)
         pixels[row * numCols + col] = 0;
   }
}

void serialDrawEdge(int *pixels, int numRows, int numCols, int edgeWidth) {
   for(int row = 0; row < numRows; row++) {
      for(int col = 0; col < numCols; col++) {
         if(col < edgeWidth || col > (numCols - edgeWidth - 1) || row < edgeWidth || row > (numRows - edgeWidth - 1))
            pixels[row * numCols + col] = 0;
      }
   }
}

__global__ void cudaDrawEdge(int *pixels, int numRows, int numCols, int edgeWidth) {
   int col = blockIdx.x * blockDim.x + threadIdx.x;
   int row = blockIdx.y * blockDim.y + threadIdx.y;

   if(col < numCols && row < numRows) {
      if(col < edgeWidth || col > (numCols - edgeWidth - 1) || row < edgeWidth || row > (numRows - edgeWidth - 1))
         pixels[row * numCols + col] = 0;
   }
}

void serialDrawLine(int *pixels, int col1, int row1, int col2, int row2, int numCols, int numRows, int bound) {    
   int num;
   for(int row = 0; row < numRows; row++) {
      for(int col = 0; col < numCols; col++) {
         // check for t in range 0 <= t <= 1
         if((row1 <= row && row <= row2) || (row1 >= row && row >= row2)) {
            // check the equality
            num = (col - col1) * (row2 - row1) - (col2 - col1) * (row - row1);
            if(-bound <= num && num <= bound)
               pixels[row * numCols + col] = 0;
         }
      }
   }
}

__global__ void cudaDrawLine(int *pixels, int col1, int row1, int col2, int row2, int numCols, int numRows, int bound) {
   int col = blockIdx.x * blockDim.x + threadIdx.x;
   int row = blockIdx.y * blockDim.y + threadIdx.y;

   int num;
   if(col < numCols && row < numRows) {
      if((row1 <= row && row <= row2) || (row1 >= row && row >= row2)) {
         num = (col - col1) * (row2 - row1) - (col2 - col1) * (row - row1);
         if(-bound <= num && num <= bound)
            pixels[row * numCols + col] = 0;
      }
   }
}
