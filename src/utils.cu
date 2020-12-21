#include "utils.h"


void getPrevDatetime(char* datetime, unsigned int secs)
{
    char* token;
    struct tm timeinfo, ptimeinfo;
    time_t timer;

    token = strtok(datetime, "-");
    timeinfo.tm_year = atoi(token) - 1900;
    token = strtok(NULL, "-");
    timeinfo.tm_mon = atoi(token) - 1;
    token = strtok(NULL, " ");
    timeinfo.tm_mday = atoi(token);

    token = strtok(NULL, ":");
    timeinfo.tm_hour = atoi(token);
    token = strtok(NULL, ":");
    timeinfo.tm_min = atoi(token);
    token = strtok(NULL, " ");
    timeinfo.tm_sec = atoi(token);

    timer = mktime(&timeinfo) - secs;       // Convert tm to time_t as local time.
    ptimeinfo = localtime(&timer);          // Convert time_t to tm as local time.
    strftime(datetime, 20, "%Y-%m-%d %H:%M:%S", ptimeinfo);
    return;
}


int cmpItdfs(const void* a, const void* b)
{
    return *(F*)a - *(F*)b >= 0 ? 1 : -1;
}
