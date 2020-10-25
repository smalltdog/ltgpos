#include <stdio.h>
#include "../src/lightning_position.h"


int main()
{
    mallocResBytes();
    printf("%s", ltgPosition("[{\"datetime\": \"2020-11-23 01:01:12\", \"longitude\": 119.8493, \"latitude\": 29.5466, \"microsecond\": 2330000, \"node\": \"waibiwaibi\", \"signal_strength\":1},"
                             "{\"datetime\": \"2020-11-23 01:01:12\", \"longitude\": 119.7540, \"latitude\": 29.4369, \"microsecond\": 8020000, \"node\": \"waibiwaibi\", \"signal_strength\":1},"
                             "{\"datetime\": \"2020-11-23 01:01:12\", \"longitude\": 119.8379, \"latitude\": 29.5471, \"microsecond\": 8700000, \"node\": \"waibiwaibi\", \"signal_strength\":1},"
                             "{\"datetime\": \"2020-11-23 01:01:12\", \"longitude\": 119.8429, \"latitude\": 29.5629, \"microsecond\": 9590000, \"node\": \"waibiwaibi\", \"signal_strength\":1}]"
                            ));
    freeResBytes();
}
