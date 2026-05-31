#ifndef IMAGE_H
#define IMAGE_H

#include <stdlib.h>
#include "renderer.h"

/* STB_IMAGE_WRITE_STATIC makes all stb functions static so this header
   can be included from multiple translation units without link errors. */
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static inline unsigned char clamp_u8(float v) {
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    return (unsigned char)(v * 255.0f + 0.5f);
}

static inline void write_png(const char* path, Image3 img, int W, int H) {
    int N = W * H;
    unsigned char* buf = (unsigned char*)malloc(N * 3);
    if (!buf) { fprintf(stderr, "write_png: out of memory\n"); return; }
    for (int i = 0; i < N; i++) {
        buf[i*3 + 0] = clamp_u8(img.r->data[i]);
        buf[i*3 + 1] = clamp_u8(img.g->data[i]);
        buf[i*3 + 2] = clamp_u8(img.b->data[i]);
    }
    if (!stbi_write_png(path, W, H, 3, buf, W * 3))
        fprintf(stderr, "write_png: failed to write %s\n", path);
    free(buf);
}

#endif
