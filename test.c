#include <stdlib.h>
#include <stdio.h>
int main() {

   system("[ ! -d test_output ] && mkdir -p test_output");

   // clear speedup file
   FILE * s = fopen("speedup.txt", "w");
   fclose(s);

   printf("\n");
   printf("RUNNING TEST EXAMPLES. THESE INCLUDE ALL CASES PUT IN THE PROMPT PLUS THE\n");
   printf("EXTRA CREDIT CASES\n\n");

   printf("====================CORRECT INPUT CASES====================\n\n");
   printf("_____________________________________________________________________\n");
   
   printf("./myPaint -c 470 355 100  ./balloons.ascii.pgm  ./test_output/balloons_c100_4.pgm\n\n");
   printf("<message>\n");
   system("./myPaint -c 470 355 100  ./balloons.ascii.pgm  ./test_output/balloons_c100_4.pgm");
   printf("</message>\n");
   printf("\n");
   printf("_____________________________________________________________________\n");
   
   printf("./myPaint -c 228 285 75  ./balloons.ascii.pgm  ./test_output/balloons_c75_5.pgm\n\n");
   printf("<message>\n");
   system("./myPaint -c 228 285 75  ./balloons.ascii.pgm  ./test_output/balloons_c75_5.pgm");
   printf("</message>\n");
   printf("\n");
   printf("_____________________________________________________________________\n");
   
   printf("./myPaint -e 50  ./balloons.ascii.pgm  ./test_output/balloons_e50_2.pgm\n\n");
   printf("<message>\n");
   system("./myPaint -e 50  ./balloons.ascii.pgm  ./test_output/balloons_e50_2.pgm");
   printf("</message>\n");
   printf("\n");
   printf("_____________________________________________________________________\n");
   
   printf("./myPaint -l  1 5 50 200  ./balloons.ascii.pgm  ./test_output/balloons_l1.pgm\n\n");
   printf("<message>\n");
   system("./myPaint -l  1 5 50 200  ./balloons.ascii.pgm  ./test_output/balloons_l1.pgm");
   printf("</message>\n");
   printf("\n");
   printf("_____________________________________________________________________\n");
   
   printf("./myPaint -l  1 50 479 639  ./balloons.ascii.pgm  ./test_output/balloons_l2.pgm\n\n");
   printf("<message>\n");
   system("./myPaint -l  1 50 479 639  ./balloons.ascii.pgm  ./test_output/balloons_l2.pgm");
   printf("</message>\n");
   printf("\n");   
   printf("_____________________________________________________________________\n");
   
   printf("./myPaint -l  479 5 0 639  ./balloons.ascii.pgm  ./test_output/balloons_l3.pgm\n\n");
   printf("<message>\n");
   system("./myPaint -l  479 5 0 639  ./balloons.ascii.pgm  ./test_output/balloons_l3.pgm");
   printf("</message>\n");
   printf("\n");
   printf("_____________________________________________________________________\n");
   
   printf("./myPaint -l   5 320 240 320  ./balloons.ascii.pgm  ./test_output/balloons_14.pgm\n\n");
   printf("<message>\n");
   system("./myPaint -l   5 320 240 320  ./balloons.ascii.pgm  ./test_output/balloons_14.pgm");
   printf("</message>\n");
   printf("\n");
   printf("_____________________________________________________________________\n");
   
   printf("./myPaint -ce 470 355 100 50 ./balloons.ascii.pgm  ./test_output/balloons_ce1.pgm\n\n");
   printf("<message>\n");
   system("./myPaint -ce 470 355 100 50 ./balloons.ascii.pgm  ./test_output/balloons_ce1.pgm");
   printf("</message>\n");
   printf("\n");
   printf("_____________________________________________________________________\n");
   
   printf("./myPaint -c -e 50 220 100 20 ./balloons.ascii.pgm  ./test_output/balloons_ce2.pgm\n\n");
   printf("<message>\n");
   system("./myPaint -c -e 50 220 100 20 ./balloons.ascii.pgm  ./test_output/balloons_ce2.pgm");
   printf("</message>\n");
   printf("\n");
   printf("_____________________________________________________________________\n\n\n\n");
   
   printf("====================INCORRECT INPUT CASES====================\n\n");
   printf("_____________________________________________________________________\n");
   
   printf("./myPaint -e 50 300  ./balloons.ascii.pgm\n\n");
   printf("<message>\n");
   system("./myPaint -e 50 300  ./balloons.ascii.pgm");
   printf("</message>\n");
   printf("\n");  
   printf("_____________________________________________________________________\n");
   
   printf("./myPaint -e ab  ./balloons.ascii.pgm\n\n");
   printf("<message>\n");
   system("./myPaint -e ab  ./balloons.ascii.pgm");
   printf("</message>\n");
   printf("\n");
   printf("_____________________________________________________________________\n");
   
   printf("./myPaint -e   ./balloons.ascii.pgm\n\n");
   printf("<message>\n");
   system("./myPaint -e   ./balloons.ascii.pgm");
   printf("</message>\n");
   printf("\n");
   printf("_____________________________________________________________________\n");
   
   printf("./myPaint -c 470 355  ./balloons.ascii.pgm\n\n");
   printf("<message>\n");
   system("./myPaint -c 470 355  ./balloons.ascii.pgm"); 
   printf("</message>\n");
   printf("\n");
   printf("_____________________________________________________________________\n");
   
   printf("./myPaint -c 470 355 50 60  ./balloons.ascii.pgm\n\n");
   printf("<message>\n");
   system("./myPaint -c 470 355 50 60  ./balloons.ascii.pgm");
   printf("</message>\n");
   printf("\n");
   printf("_____________________________________________________________________\n");
   
   printf("./myPaint -c 470 90bc   ./balloons.ascii.pgm\n\n");
   printf("<message>\n");
   system("./myPaint -c 470 90bc   ./balloons.ascii.pgm");
   printf("</message>\n");
   printf("\n");
   printf("_____________________________________________________________________\n\n");
   
   printf("Check 'test_output' directory for output pgm files\n\n");
   printf("Note: the extra credit was implemented where both a circle and a border could\n");
   printf("be placed at the same time. Two test example where shown for both syntax cases.\n\n");
	return 0;
}
