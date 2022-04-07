#include "pgmUtility.h"
#include "pgmProcess.cu"

/* READ PIXELS INTO PIXEL ARRAY */
int * pgmRead(char ** header, int * numRows, int * numCols, FILE *in) {
        for(int i = 0; i < rowsInHeader; i++) {
                if(fgets(header[i], maxSizeHeadRow, in) == NULL) 
                        return NULL;
        }

        // rows and cols of pixels
        sscanf(header[2], "%d  %d", numCols, numRows);
   
        // init pixel array
        int *pixels = (int *)malloc((*numRows) * (*numCols) * sizeof(int));

        // read pixels into pixel array
        int r, c;
        for(r = 0; r < *numRows; r++) {
                for(c = 0; c < *numCols; c++) {
                        if(fscanf(in, "%d ", &pixels[r * (*numCols) + c]) < 0)
                                return NULL;
                }
        }

        return pixels;
}

int pgmDrawCircle( int *pixels, int numRows, int numCols, int centerRow, int centerCol, int radius, char **header) {
        // run on cpu
        double start = currentTime();
        serialDrawCircle(pixels, numCols, numRows, centerCol, centerRow, radius);
        double end = currentTime();
        double time = end - start;

        // set up cuda
        int * cuda_pixels;
        int n = numRows * numCols + 1;
        size_t bytes = n * sizeof(int);
        cudaMalloc(&cuda_pixels, bytes);
        cudaMemcpy(cuda_pixels, pixels, bytes, cudaMemcpyHostToDevice);
	dim3 grid, block;
	block.x = 16;
	block.y = 16;
	grid.x = (int)ceil((float)numCols / block.x);
	grid.y = (int)ceil((float)numRows / block.y);
        
        // run on gpu
        double pstart = currentTime();
        cudaDrawCircle<<<grid, block>>>(cuda_pixels, numCols, numRows, centerCol, centerRow, radius);
        double pend = currentTime();
        double ptime = pend - pstart;
        
        // extract and free gpu resources
        cudaMemcpy(pixels, cuda_pixels, bytes, cudaMemcpyDeviceToHost);
        cudaFree(cuda_pixels);
 
	char name[7] = { 'C', 'i', 'r', 'c', 'l', 'e', '\0' };       
        printSpeedup(time, ptime, name); 
        return recalculateIntensity(pixels, header, numCols, numRows);
}

int pgmDrawEdge( int *pixels, int numRows, int numCols, int edgeWidth, char **header) {
        
        // run cpu
        double start = currentTime();
        serialDrawEdge(pixels, numRows, numCols, edgeWidth);
        double end = currentTime();
        double time = end - start;

        // set up cuda
        int * cuda_pixels;
        int n = numRows * numCols + 1;
        size_t bytes = n * sizeof(int);
        cudaMalloc(&cuda_pixels, bytes);
        cudaMemcpy(cuda_pixels, pixels, bytes, cudaMemcpyHostToDevice);
	dim3 grid, block;
	block.x = 16;
	block.y = 16;
	grid.x = (int)ceil((float)numCols / block.x);
	grid.y = (int)ceil((float)numRows / block.y);
        
        // Execute on gpu
        double pstart = currentTime();
        cudaDrawEdge<<<grid, block>>>(cuda_pixels, numRows, numCols, edgeWidth);
        double pend = currentTime();
        double ptime = pend - pstart;

        // extract and free gpu resources        
        cudaMemcpy(pixels, cuda_pixels, bytes, cudaMemcpyDeviceToHost);
        cudaFree(cuda_pixels);

	char name[5] = { 'E', 'd', 'g', 'e', '\0' };
        printSpeedup(time, ptime, name);
        return recalculateIntensity(pixels, header, numCols, numRows);
}

int pgmDrawLine(int *pixels, int numRows, int numCols, char **header, int p1row, int p1col, int p2row, int p2col) {
        /*
            The vector equation of a line segment is:
            <col, row> = (1-t)<col1, row1> + t<col2, row2>
            for 0 <= t <= 1
            
            Combining the vectors gives:
            <col, row> = < col1+t(-col1 + col2), row1+t(-row1 + row2) >
            
            Breaking into component-wise equations gives:
            col = col1+t(-col1 + col2)
            row = row1+t(-row1 + row2)
            
            Which, when isolating for t, become:
            t = (col - col1)/(col2 - col1)
            t = (row - row1)/(row2 - row1)
         
            Therefore, the point (col, row) is on the line segment if these
            equations for t are equal, and if t is within bounds. The test 
            for equalisty is:
            
            (col - col1)/(col2 - col1) = (row - row1)/(row2 - row1)
        <=> (col - col1)(row2 - row1) = (col2 - col1)(row - row1)
         
            And the test for bounds (arbitrarily using row, either one could be used):
            0 <= (row - row1)/(row2 - row1) <= 1
        <=> (0 <= row - row1 <= row2 - row1)  OR  (0 >= row - row1 >= row2 - row1)  [in case the denom was neg]
        <=> [row1 <= row <= row2]  OR  [row1 >= row >= row2]
         
            The code will first test for the bounds question, then will test the equality.
                        
            The test for equality is perfect for the real number plane, but the discrete
            pixels will likely not be 'perfectly' enough aligned, so we must alter it to
            allow for the natrual error. Lets say I wanted to do all pixels that are a
            distance of 1 row pixel from the perfect line.
            
            (col - col1)(row2 - row1) = (col2 - col1)(row - row1)
        <=> (col - col1)(row2 - row1) - (col2 - col1)(row - row1) = 0
         => (col - col1)(row2 - row1) - (col2 - col1)(row += 1- row1) = 0
        <=> (col - col1)(row2 - row1) - (col2 - col1)(row - row1) +- (col2 - col1) = 0
        <=> (col - col1)(row2 - row1) - (col2 - col1)(row - row1) = +- (col2 - col1)
        
            If I wanted to  do this for 1 column pixel instead the same logic would give:
            
            (col - col1)(row2 - row1) - (col2 - col1)(row - row1) = +- (row2 - row1)
            
            If I wanted to fill in a line of all pixels within the bounds of the min and max of the
            right side that would create the following inequalities:
            
            |(col - col1)(row2 - row1) - (col2 - col1)(row - row1)| <= |col2 - col1|
            |(col - col1)(row2 - row1) - (col2 - col1)(row - row1)| <= |row2 - row1|
            
            One uses the bounds for a row pixel, one for a column pixel. To get a decent value, I'll 
            simply average the two:
            
            |(col - col1)(row2 - row1) - (col2 - col1)(row - row1)| <= (|row2 - row1| + |col2 - col1|)/2
            
            This will serve as the equality test for a line. 
        */
        
        // Set up bound according the math just discussed
        int col_pixel = p2row - p1row;
        if(col_pixel < 0)
           col_pixel *= -1;
      
        int row_pixel = p2col - p1col;
        if(row_pixel < 0)
           row_pixel *= -1;
      
        int bound = (int)((col_pixel + row_pixel) / 2);
        
        // Run on cpu
        double start = currentTime();
        serialDrawLine(pixels, p1col, p1row, p2col, p2row, numCols, numRows, bound);
        double end = currentTime();
        double time = end - start;

        // set up cuda
        int * cuda_pixels;
        int n = numRows * numCols + 1;
        size_t bytes = n * sizeof(int);
        cudaMalloc(&cuda_pixels, bytes);
        cudaMemcpy(cuda_pixels, pixels, bytes, cudaMemcpyHostToDevice);
	dim3 grid, block;
	block.x = 16;
	block.y = 16;
	grid.x = ceil((float)numCols / block.x);
	grid.y = ceil((float)numRows / block.y);

        // run on gpu
        double pstart = currentTime();
        cudaDrawLine<<<grid, block>>>(cuda_pixels, p1col, p1row, p2col, p2row, numCols, numRows, bound);
        double pend = currentTime();
        double ptime = pend - pstart;
        
        // extract and free gpu resources
        cudaMemcpy(pixels, cuda_pixels, bytes, cudaMemcpyDeviceToHost);
        cudaFree(cuda_pixels);

	char name[5] = { 'L', 'i', 'n', 'e', '\0'};
        printSpeedup(time, ptime, name);
        return recalculateIntensity(pixels, header, numCols, numRows);
}

int pgmWrite(char **header, const int *pixels, int numRows, int numCols, FILE *out) {
        int i, j;

        // write the header
        for ( i = 0; i < rowsInHeader; i ++ ) {
                fprintf(out, "%s", *( header + i ) );
        }
        fprintf(out, "%s", "\n");

        // write the pixels
        for( i = 0; i < numRows; i ++ ) {
                for ( j = 0; j < numCols; j ++ ) {
                        if ( j < numCols - 1 )
                                fprintf(out, "%d ", pixels[i * numCols + j]);
                        else
                                fprintf(out, "%d\n", pixels[i * numCols + j]);
                }
        }
        return 0;
}

int recalculateIntensity(int * pixels, char ** header, int numCols, int numRows) {
        int intensity = 0;
        for(int i = 0; i < numCols * numRows - 1; i++) {
                if(intensity < pixels[i])
                        intensity = pixels[i];
        }

        if(intensity == atoi(header[3])) {
                return 0;
        } else {
                sprintf(header[3], "%d", intensity);
                return 1;
        }
}

void printSpeedup(double time, double ptime, char * name) {
   FILE * out = fopen("speedup.txt", "a");
   fprintf(out, "Draw %s:\n", name);
   fprintf(out, "%s %f\n", "Serial Time was", time);
   fprintf(out, "%s %f\n", "Parallel Time was", ptime);
   fprintf(out, "%s %f\n\n", "Speedup factor was", time / ptime);
   fclose(out);
}

double currentTime(){
   struct timeval now;
   gettimeofday(&now, NULL);

   return now.tv_sec + now.tv_usec/1000000.0;
}
