#include "pgmUtility.cu"

void usage();

int main(int argc, char *argv[]) {
   // Make sure at least 2 arguments are included
   // this lets me check flags without error
   if(argc < 2) {
      usage();
      return 1;
   }
   
   // Check flag and length, then get choice and filenames
   char choice;
   char * inputname;
   char * outputname;
   
   if(strcmp(argv[1], "-ce") == 0) { // first extra credit case
      if(argc != 8) {
         usage();
         return 1;
      }
      
      // I went with 't' for two
      choice = 't';
      inputname = argv[6];
      outputname = argv[7];
   } else if(strcmp(argv[1], "-c") == 0 && strcmp(argv[2], "-e") == 0) { // second extra credit case
      if(argc != 9) {
         usage();
         return 1;
      }
      
      choice = 't';
      inputname = argv[7];
      outputname = argv[8];
   } else if(strcmp(argv[1], "-c") == 0) {
      if(argc != 7) {
         usage();
         return 1;
      }
      
      choice = 'c';
      inputname = argv[5];
      outputname = argv[6];
   } else if(strcmp(argv[1], "-e") == 0) {
      if(argc != 5) {
         usage();
         return 1;
      }
      
      choice = 'e';
      inputname = argv[3];
      outputname = argv[4];
   } else if(strcmp(argv[1], "-l") == 0) {
      if(argc != 8) {
         usage();
         return 1;
      }
      
      choice = 'l';
      inputname = argv[6];
      outputname = argv[7];
   } else {
      usage();
      return 1;
   }
   
   // Init pgm elements
   char ** header = (char**) malloc( sizeof(char *) * rowsInHeader);
   for(int i = 0; i < rowsInHeader; i++)
       header[i] = (char *) malloc (sizeof(char) * maxSizeHeadRow);
   
   int numRows, numCols;
   int * pixels = NULL;

   // Open and read input file into pgm elements
   FILE * in = fopen(inputname, "r");
   if(in == NULL) {
      printf("invalid input filename (no file named '%s' found)\n", inputname);
      usage();
      return 1;
   }
   
   pixels = pgmRead(header, &numRows, &numCols, in);
   fclose(in);
   
   // process file alterations
   if(choice == 't') {
      int centerRow, centerCol, radius, edgeWidth;
      if(strcmp(argv[1], "-ce") == 0) {
         centerRow = atoi(argv[2]);
         centerCol = atoi(argv[3]);
         radius = atoi(argv[4]);
         edgeWidth = atoi(argv[5]);
      } else {
         centerRow = atoi(argv[3]);
         centerCol = atoi(argv[4]);
         radius = atoi(argv[5]);
         edgeWidth = atoi(argv[6]);
      }
            
      pgmDrawEdge(pixels, numRows, numCols, edgeWidth, header);
      pgmDrawCircle(pixels, numRows, numCols, centerRow, centerCol, radius, header);
   } else if(choice == 'c') {
      // Get specific variables
      int centerRow = atoi(argv[2]);
      int centerCol = atoi(argv[3]);
      int radius = atoi(argv[4]);
      
      pgmDrawCircle(pixels, numRows, numCols, centerRow, centerCol, radius, header);
   } else if(choice == 'e') {
      // Get specific variables
      int edgeWidth = atoi(argv[2]);
      
      pgmDrawEdge(pixels, numRows, numCols, edgeWidth, header);
   } else { // choice == 'l'
      // Get specific variables
      int p1row = atoi(argv[2]);
      int p1col = atoi(argv[3]);
      int p2row = atoi(argv[4]);
      int p2col = atoi(argv[5]);
      
      pgmDrawLine(pixels, numRows, numCols, header, p1row, p1col, p2row, p2col);
   }
   
   // Write output file
   FILE * out = fopen(outputname, "w");
   pgmWrite(header, pixels, numRows, numCols, out);
   fclose(out);

   // Free Memory
   free(pixels);
   for(int i = 0; i < rowsInHeader; i++)
      free(header[i]);
   free(header);
   
   return 0;
}

void usage() {
   printf("Usage:\n");
   printf("\t-e edgeWidth  oldImageFile  newImageFile\n");
   printf("\t-c circleCenterRow circleCenterCol radius  oldImageFile  newImageFile\n");
   printf("\t-l  p1row  p1col  p2row  p2col  oldImageFile  newImageFile\n");
}
