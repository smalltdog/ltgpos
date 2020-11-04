#include <stdio.h>
#include "../../src/lightning_position.h"


int main()
{
    mallocResBytes();
    setCfg(20, 100 * 100 * 100, 2.2, 1 / C, true);
    for (int i = 0; i < 2; ++i) {
        printf("%s\n\n\n",
               ltgPosition("[{\"datetime\": \"2020-11-23 01:01:12\", \"longitude\": 103.48899999, \"latitude\": 31.0612, \"microsecond\": 7366123, \"node\": \"1\", \"signal_strength\":1},"
                           "{\"datetime\": \"2020-11-23 01:01:12\", \"longitude\": 105.7179999999, \"latitude\": 33.7411, \"microsecond\": 7373554, \"node\": \"7\", \"signal_strength\":1},"
                           "{\"datetime\": \"2020-11-23 01:01:12\", \"longitude\": 105.9029999999, \"latitude\": 34.5542, \"microsecond\": 7376610, \"node\": \"10\", \"signal_strength\":1}]"
        ));
        setCfg(20, 80 * 80 * 80, 2.2, 0.5 / C, false);
    }
    freeResBytes();
}
