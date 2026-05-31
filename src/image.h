#ifndef IMAGE_H
#define IMAGE_H

#include <stdio.h>
#include "renderer.h"

static inline unsigned char clamp_u8(float v) {
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    return (unsigned char)(v * 255.0f + 0.5f);
}

/* binary P6 PPM */
static inline void write_ppm(const char* path, Image3 img, int W, int H) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "write_ppm: cannot open %s\n", path); return; }
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    int N = W * H;
    for (int i = 0; i < N; i++) {
        fputc(clamp_u8(img.r->data[i]), f);
        fputc(clamp_u8(img.g->data[i]), f);
        fputc(clamp_u8(img.b->data[i]), f);
    }
    fclose(f);
}

#endif
