#include <stdio.h>

#include "../src/ltgpos.h"


int main(int argc, char** argv)
{
    char buf[10240];
    char* filename = argv[1];
    FILE* fp = fopen(filename, "r");

    if (!fp) {
        fprintf(stderr, "%s(%d): failed to open file %s.\n",
                __FILE__, __LINE__, filename);
        return 1;
    }

    int i = 0;
    while (!feof(fp)) {
        if (!fgets(buf, sizeof(buf), fp)) break;
        ltgpos(buf);
        i++;
    }

    freeSysInfo();
    fclose(fp);
    return 0;
}
