#include "waveclf.h"


int main()
{
    clf = WaveClf("asd");
    clf.predict(2, vector<float>({1, 2, 3}));
    return 0;
}
