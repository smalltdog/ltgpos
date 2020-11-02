#include <stdio.h>
#include "../../src/lightning_position.h"


int main()
{
    mallocResBytes();
    for (int i = 0; i < 2; ++i) {
        printf("%s\n\n\n",
            ltgPosition("[{\"datetime\": \"2020-11-23 01:01:12\", \"longitude\": 119.7008, \"latitude\": 29.1371, \"microsecond\": 8709720, \"node\": \"w\", \"signal_strength\":1},"
                        "{\"datetime\": \"2020-11-23 01:01:12\", \"longitude\": 119.2652, \"latitude\": 29.4752, \"microsecond\": 8710023, \"node\": \"wa\", \"signal_strength\":1},"
                        "{\"datetime\": \"2020-11-23 01:01:12\", \"longitude\": 119.6792, \"latitude\": 30.202, \"microsecond\": 8710633, \"node\": \"wa\", \"signal_strength\":1},"
                        "{\"datetime\": \"2020-11-23 01:01:12\", \"longitude\": 120.8962, \"latitude\": 29.5043, \"microsecond\": 8711582, \"node\": \"wa\", \"signal_strength\":1},"
                        "{\"datetime\": \"2020-11-23 01:01:12\", \"longitude\": 118.9142, \"latitude\": 28.9434, \"microsecond\": 8711896, \"node\": \"wa\", \"signal_strength\":1},"
                        "{\"datetime\": \"2020-11-23 01:01:12\", \"longitude\": 120.7407, \"latitude\": 28.8681, \"microsecond\": 8712021, \"node\": \"wa\", \"signal_strength\":1},"
                        "{\"datetime\": \"2020-11-23 01:01:12\", \"longitude\": 120.08, \"latitude\": 30.9083, \"microsecond\": 8713253, \"node\": \"wa\", \"signal_strength\":1},"
                        "{\"datetime\": \"2020-11-23 01:01:12\", \"longitude\": 121.2862, \"latitude\": 30.1702, \"microsecond\": 8713352, \"node\": \"wa\", \"signal_strength\":1},"
                        "{\"datetime\": \"2020-11-23 01:01:12\", \"longitude\": 121.4389, \"latitude\": 28.6375, \"microsecond\": 8714332, \"node\": \"wa\", \"signal_strength\":1},"
                        "{\"datetime\": \"2020-11-23 01:01:12\", \"longitude\": 121.1224, \"latitude\": 31.1551, \"microsecond\": 8715378, \"node\": \"wa\", \"signal_strength\":1}]"));
        setCfg(20, 80 * 80 * 80, 1.2, 1 / C, false);
    }
    freeResBytes();
}
