#include <stdio.h>

#include "../src/ltgpos.h"


int main(int argc, char** argv)
{
    char buf[2048];
    char* filename = argv[1];
    FILE* fp = fopen(filename, "r");

    if (!fp) {
        fprintf(stderr, "%s(%d): failed to open file %s.\n",
                __FILE__, __LINE__, filename);
        return 1;
    }

    initSysInfo();

    int i = 0;
    while (!feof(fp)) {
        if (i == 1) break;
        fgets(buf, sizeof(buf), fp);
        ltgpos(buf);
        printf("\n================================\n\n");
        i++;
    }

    freeSysInfo();
    fclose(fp);
    return 0;
}
