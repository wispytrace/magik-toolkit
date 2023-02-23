/*
 *        (C) COPYRIGHT Ingenic Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : drawing.hpp
 * Authors    : lqwang
 * Create Time: 2022-03-02:11:18:25
 * Description:
 *
 */

#include <cassert>
#include <cfloat>
#include "common_data_struct.h"


enum { XY_SHIFT = 16, XY_ONE = 1 << XY_SHIFT, DRAWING_STORAGE_BLOCK = (1<<12) - 256 };

enum { SAMPLE_FONT_SIZE_SHIFT=8, SAMPLE_FONT_ITALIC_ALPHA=(1 << 8),
       SAMPLE_FONT_ITALIC_DIGIT=(2 << 8), SAMPLE_FONT_ITALIC_PUNCT=(4 << 8),
       SAMPLE_FONT_ITALIC_BRACES=(8 << 8), SAMPLE_FONT_HAVE_GREEK=(16 << 8),
       SAMPLE_FONT_HAVE_CYRILLIC=(32 << 8) };

static const int HersheyPlain[] = {
(5 + 4*16) + SAMPLE_FONT_HAVE_GREEK,
199, 214, 217, 233, 219, 197, 234, 216, 221, 222, 228, 225, 211, 224, 210, 220,
200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 212, 213, 191, 226, 192,
215, 190, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 193, 84,
194, 85, 86, 87, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
195, 223, 196, 88 };

static const int HersheyPlainItalic[] = {
(5 + 4*16) + SAMPLE_FONT_ITALIC_ALPHA + SAMPLE_FONT_HAVE_GREEK,
199, 214, 217, 233, 219, 197, 234, 216, 221, 222, 228, 225, 211, 224, 210, 220,
200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 212, 213, 191, 226, 192,
215, 190, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 193, 84,
194, 85, 86, 87, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161,
162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176,
195, 223, 196, 88 };

static const int HersheyComplexSmall[] = {
(6 + 7*16) + SAMPLE_FONT_HAVE_GREEK,
1199, 1214, 1217, 1275, 1274, 1271, 1272, 1216, 1221, 1222, 1219, 1232, 1211, 1231, 1210, 1220,
1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1212, 2213, 1241, 1238, 1242,
1215, 1273, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013,
1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1223, 1084,
1224, 1247, 586, 1249, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111,
1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126,
1225, 1229, 1226, 1246 };

static const int HersheyComplexSmallItalic[] = {
(6 + 7*16) + SAMPLE_FONT_ITALIC_ALPHA + SAMPLE_FONT_HAVE_GREEK,
1199, 1214, 1217, 1275, 1274, 1271, 1272, 1216, 1221, 1222, 1219, 1232, 1211, 1231, 1210, 1220,
1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1212, 1213, 1241, 1238, 1242,
1215, 1273, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063,
1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1223, 1084,
1224, 1247, 586, 1249, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161,
1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176,
1225, 1229, 1226, 1246 };

static const int HersheySimplex[] = {
(9 + 12*16) + SAMPLE_FONT_HAVE_GREEK,
2199, 714, 717, 733, 719, 697, 734, 716, 721, 722, 728, 725, 711, 724, 710, 720,
700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 712, 713, 691, 726, 692,
715, 690, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513,
514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 693, 584,
694, 2247, 586, 2249, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611,
612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626,
695, 723, 696, 2246 };

static const int HersheyDuplex[] = {
(9 + 12*16) + SAMPLE_FONT_HAVE_GREEK,
2199, 2714, 2728, 2732, 2719, 2733, 2718, 2727, 2721, 2722, 2723, 2725, 2711, 2724, 2710, 2720,
2700, 2701, 2702, 2703, 2704, 2705, 2706, 2707, 2708, 2709, 2712, 2713, 2730, 2726, 2731,
2715, 2734, 2501, 2502, 2503, 2504, 2505, 2506, 2507, 2508, 2509, 2510, 2511, 2512, 2513,
2514, 2515, 2516, 2517, 2518, 2519, 2520, 2521, 2522, 2523, 2524, 2525, 2526, 2223, 2084,
2224, 2247, 587, 2249, 2601, 2602, 2603, 2604, 2605, 2606, 2607, 2608, 2609, 2610, 2611,
2612, 2613, 2614, 2615, 2616, 2617, 2618, 2619, 2620, 2621, 2622, 2623, 2624, 2625, 2626,
2225, 2229, 2226, 2246 };

static const int HersheyComplex[] = {
(9 + 12*16) + SAMPLE_FONT_HAVE_GREEK + SAMPLE_FONT_HAVE_CYRILLIC,
2199, 2214, 2217, 2275, 2274, 2271, 2272, 2216, 2221, 2222, 2219, 2232, 2211, 2231, 2210, 2220,
2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2212, 2213, 2241, 2238, 2242,
2215, 2273, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,
2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2223, 2084,
2224, 2247, 587, 2249, 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111,
2112, 2113, 2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121, 2122, 2123, 2124, 2125, 2126,
2225, 2229, 2226, 2246, 2801, 2802, 2803, 2804, 2805, 2806, 2807, 2808, 2809, 2810, 2811,
2812, 2813, 2814, 2815, 2816, 2817, 2818, 2819, 2820, 2821, 2822, 2823, 2824, 2825, 2826,
2827, 2828, 2829, 2830, 2831, 2832, 2901, 2902, 2903, 2904, 2905, 2906, 2907, 2908, 2909,
2910, 2911, 2912, 2913, 2914, 2915, 2916, 2917, 2918, 2919, 2920, 2921, 2922, 2923, 2924,
2925, 2926, 2927, 2928, 2929, 2930, 2931, 2932};

static const int HersheyComplexItalic[] = {
(9 + 12*16) + SAMPLE_FONT_ITALIC_ALPHA + SAMPLE_FONT_ITALIC_DIGIT + SAMPLE_FONT_ITALIC_PUNCT +
SAMPLE_FONT_HAVE_GREEK + SAMPLE_FONT_HAVE_CYRILLIC,
2199, 2764, 2778, 2782, 2769, 2783, 2768, 2777, 2771, 2772, 2219, 2232, 2211, 2231, 2210, 2220,
2750, 2751, 2752, 2753, 2754, 2755, 2756, 2757, 2758, 2759, 2212, 2213, 2241, 2238, 2242,
2765, 2273, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059, 2060, 2061, 2062, 2063,
2064, 2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2223, 2084,
2224, 2247, 587, 2249, 2151, 2152, 2153, 2154, 2155, 2156, 2157, 2158, 2159, 2160, 2161,
2162, 2163, 2164, 2165, 2166, 2167, 2168, 2169, 2170, 2171, 2172, 2173, 2174, 2175, 2176,
2225, 2229, 2226, 2246 };

static const int HersheyTriplex[] = {
(9 + 12*16) + SAMPLE_FONT_HAVE_GREEK,
2199, 3214, 3228, 3232, 3219, 3233, 3218, 3227, 3221, 3222, 3223, 3225, 3211, 3224, 3210, 3220,
3200, 3201, 3202, 3203, 3204, 3205, 3206, 3207, 3208, 3209, 3212, 3213, 3230, 3226, 3231,
3215, 3234, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013,
2014, 3015, 3016, 3017, 3018, 3019, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 2223, 2084,
2224, 2247, 587, 2249, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111,
3112, 3113, 3114, 3115, 3116, 3117, 3118, 3119, 3120, 3121, 3122, 3123, 3124, 3125, 3126,
2225, 2229, 2226, 2246 };

static const int HersheyTriplexItalic[] = {
(9 + 12*16) + SAMPLE_FONT_ITALIC_ALPHA + SAMPLE_FONT_ITALIC_DIGIT +
SAMPLE_FONT_ITALIC_PUNCT + SAMPLE_FONT_HAVE_GREEK,
2199, 3264, 3278, 3282, 3269, 3233, 3268, 3277, 3271, 3272, 3223, 3225, 3261, 3224, 3260, 3270,
3250, 3251, 3252, 3253, 3254, 3255, 3256, 3257, 3258, 3259, 3262, 3263, 3230, 3226, 3231,
3265, 3234, 3051, 3052, 3053, 3054, 3055, 3056, 3057, 3058, 3059, 3060, 3061, 3062, 3063,
2064, 3065, 3066, 3067, 3068, 3069, 3070, 3071, 3072, 3073, 3074, 3075, 3076, 2223, 2084,
2224, 2247, 587, 2249, 3151, 3152, 3153, 3154, 3155, 3156, 3157, 3158, 3159, 3160, 3161,
3162, 3163, 3164, 3165, 3166, 3167, 3168, 3169, 3170, 3171, 3172, 3173, 3174, 3175, 3176,
2225, 2229, 2226, 2246 };

static const int HersheyScriptSimplex[] = {
(9 + 12*16) + SAMPLE_FONT_ITALIC_ALPHA + SAMPLE_FONT_HAVE_GREEK,
2199, 714, 717, 733, 719, 697, 734, 716, 721, 722, 728, 725, 711, 724, 710, 720,
700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 712, 713, 691, 726, 692,
715, 690, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563,
564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 693, 584,
694, 2247, 586, 2249, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661,
662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676,
695, 723, 696, 2246 };

static const int HersheyScriptComplex[] = {
(9 + 12*16) + SAMPLE_FONT_ITALIC_ALPHA + SAMPLE_FONT_ITALIC_DIGIT + SAMPLE_FONT_ITALIC_PUNCT + SAMPLE_FONT_HAVE_GREEK,
2199, 2764, 2778, 2782, 2769, 2783, 2768, 2777, 2771, 2772, 2219, 2232, 2211, 2231, 2210, 2220,
2750, 2751, 2752, 2753, 2754, 2755, 2756, 2757, 2758, 2759, 2212, 2213, 2241, 2238, 2242,
2215, 2273, 2551, 2552, 2553, 2554, 2555, 2556, 2557, 2558, 2559, 2560, 2561, 2562, 2563,
2564, 2565, 2566, 2567, 2568, 2569, 2570, 2571, 2572, 2573, 2574, 2575, 2576, 2223, 2084,
2224, 2247, 586, 2249, 2651, 2652, 2653, 2654, 2655, 2656, 2657, 2658, 2659, 2660, 2661,
2662, 2663, 2664, 2665, 2666, 2667, 2668, 2669, 2670, 2671, 2672, 2673, 2674, 2675, 2676,
2225, 2229, 2226, 2246 };

static const uint8_t SlopeCorrTable[] = {
    181, 181, 181, 182, 182, 183, 184, 185, 187, 188, 190, 192, 194, 196, 198, 201,
    203, 206, 209, 211, 214, 218, 221, 224, 227, 231, 235, 238, 242, 246, 250, 254
};

static const int FilterTable[] = {
    168, 177, 185, 194, 202, 210, 218, 224, 231, 236, 241, 246, 249, 252, 254, 254,
    254, 254, 252, 249, 246, 241, 236, 231, 224, 218, 210, 202, 194, 185, 177, 168,
    158, 149, 140, 131, 122, 114, 105, 97, 89, 82, 75, 68, 62, 56, 50, 45,
    40, 36, 32, 28, 25, 22, 19, 16, 14, 12, 11, 9, 8, 7, 5, 5
};

enum LineTypes {
    FILLED  = -1,
    LINE_4  = 4, //!< 4-connected line
    LINE_8  = 8, //!< 8-connected line
    LINE_AA = 16 //!< antialiased line
};

static int _max(int a, int b)
{
    return a > b ? a : b;
}

static int _min(int a, int b)
{
    return a > b ? b : a;
}



static inline void ICV_HLINE_X(uint8_t* ptr, int xl, int xr,
                               const uint8_t* color, int pix_size)
{
    uint8_t* hline_min_ptr = ptr + (xl)*(pix_size);
    uint8_t* hline_end_ptr = ptr + (xr+1)*(pix_size);
    uint8_t* hline_ptr = hline_min_ptr;
    if (pix_size == 1)
        memset(hline_min_ptr, *color, hline_end_ptr - hline_min_ptr);
    else {//pix_size != 1
        if(hline_min_ptr < hline_end_ptr) {
            memcpy(hline_ptr, color, pix_size);
            hline_ptr += pix_size;
      }
        int sizeToCopy = pix_size;
        while(hline_ptr < hline_end_ptr) {
            memcpy(hline_ptr, hline_min_ptr, sizeToCopy);
            hline_ptr += sizeToCopy;
            sizeToCopy = _min(2*sizeToCopy, hline_end_ptr - hline_ptr);
        }
    }
}

static inline void ICV_HLINE(uint8_t* ptr, int xl, int xr, const void* color, int pix_size)
{
    ICV_HLINE_X(ptr, xl, xr, (uint8_t*)color, pix_size);
}

static int clipLine( int64_t width, int64_t height, Point pt1, Point pt2 )
{
    int c1, c2;
    int64_t right = width-1, bottom = height-1;

    if( width <= 0 || height <= 0 )
        return 0;

    int64_t x1 = pt1.x, y1 = pt1.y, x2 = pt2.x, y2 = pt2.y;
    c1 = (x1 < 0) + (x1 > right) * 2 + (y1 < 0) * 4 + (y1 > bottom) * 8;
    c2 = (x2 < 0) + (x2 > right) * 2 + (y2 < 0) * 4 + (y2 > bottom) * 8;

    if( (c1 & c2) == 0 && (c1 | c2) != 0 )
    {
        int64_t a;
        if( c1 & 12 )
        {
            a = c1 < 8 ? 0 : bottom;
            x1 += (int64_t)((float)(a - y1) * (x2 - x1) / (y2 - y1));
            y1 = a;
            c1 = (x1 < 0) + (x1 > right) * 2;
        }
        if( c2 & 12 )
        {
            a = c2 < 8 ? 0 : bottom;
            x2 += (int64_t)((float)(a - y2) * (x2 - x1) / (y2 - y1));
            y2 = a;
            c2 = (x2 < 0) + (x2 > right) * 2;
        }
        if( (c1 & c2) == 0 && (c1 | c2) != 0 )
        {
            if( c1 )
            {
                a = c1 == 1 ? 0 : right;
                y1 += (int64_t)((float)(a - x1) * (y2 - y1) / (x2 - x1));
                x1 = a;
                c1 = 0;
            }
            if( c2 )
            {
                a = c2 == 1 ? 0 : right;
                y2 += (int64_t)((float)(a - x2) * (y2 - y1) / (x2 - x1));
                x2 = a;
                c2 = 0;
            }
        }

        assert( (c1 & c2) != 0 || (x1 | y1 | x2 | y2) >= 0 );
    }

    return (c1 | c2) == 0;
}

static void Line2( Img* img, Point pt1, Point pt2, const void* color)
{
    int64_t dx, dy;
    int ecount;
    int64_t ax, ay;
    int64_t i, j;
    int x, y;
    int64_t x_step, y_step;
    uint8_t *ptr = (uint8_t*)(img->data), *tptr;
    int width = img->w;
    int height = img->h;
    int step = img->w_stride;
    if( !clipLine(width<<XY_SHIFT, height<<XY_SHIFT, pt1, pt2) )
        return;
    dx = pt2.x - pt1.x;
    dy = pt2.y - pt1.y;

    j = dx < 0 ? -1 : 0;
    ax = (dx ^ j) - j;
    i = dy < 0 ? -1 : 0;
    ay = (dy ^ i) - i;

    if( ax > ay )
    {
        dy = (dy ^ j) - j;
        pt1.x ^= pt2.x & j;
        pt2.x ^= pt1.x & j;
        pt1.x ^= pt2.x & j;
        pt1.y ^= pt2.y & j;
        pt2.y ^= pt1.y & j;
        pt1.y ^= pt2.y & j;

        x_step = XY_ONE;
        y_step = (dy << XY_SHIFT) / (ax | 1);
        ecount = (int)((pt2.x - pt1.x) >> XY_SHIFT);
    }
    else
    {
        dx = (dx ^ i) - i;
        pt1.x ^= pt2.x & i;
        pt2.x ^= pt1.x & i;
        pt1.x ^= pt2.x & i;
        pt1.y ^= pt2.y & i;
        pt2.y ^= pt1.y & i;
        pt1.y ^= pt2.y & i;

        x_step = (dx << XY_SHIFT) / (ay | 1);
        y_step = XY_ONE;
        ecount = (int)((pt2.y - pt1.y) >> XY_SHIFT);
    }

    pt1.x += (XY_ONE >> 1);
    pt1.y += (XY_ONE >> 1);

#define  ICV_PUT_POINT(_x,_y)                   \
    x = (_x); y = (_y);                         \
    if( 0 <= x && x < width &&                  \
        0 <= y && y < height )                  \
    {                                           \
        tptr = ptr + y*step + x*img->c;      \
        for( j = 0; j < img->c; j++ )        \
            tptr[j] = ((uint8_t*)color)[j];     \
    }

    ICV_PUT_POINT((int)((pt2.x + (XY_ONE >> 1)) >> XY_SHIFT),
                  (int)((pt2.y + (XY_ONE >> 1)) >> XY_SHIFT));

    if( ax > ay )
    {
        pt1.x >>= XY_SHIFT;

        while( ecount >= 0 )
        {
            ICV_PUT_POINT((int)(pt1.x), (int)(pt1.y >> XY_SHIFT));
            pt1.x++;
            pt1.y += y_step;
            ecount--;
        }
    }
    else
    {
        pt1.y >>= XY_SHIFT;

        while( ecount >= 0 )
        {
            ICV_PUT_POINT((int)(pt1.x >> XY_SHIFT), (int)(pt1.y));
            pt1.x += x_step;
            pt1.y++;
            ecount--;
        }
    }

#undef ICV_PUT_POINT
}

static void FillConvexPoly(Img* img, const Point* v, int npts,
                           const void* color, int line_type, int shift)
{
    struct{
        int idx, di;
        int64_t x,   dx;
        int ye;
    } edge[2];
    int delta = 1 << shift >> 1;
    int i, y, imin = 0;
    int edges = npts;
    int64_t xmin, xmax, ymin, ymax;
    uint8_t* ptr = (uint8_t*)(img->data);
    int width  = img->w;
    int height = img->h;
    Point p0;
    int delta1, delta2;
    if(line_type < LINE_AA)
        delta1 = delta2 = XY_ONE >> 1;
    else
        delta1 = XY_ONE - 1, delta2 = 0;
    p0 = v[npts - 1];
    p0.x <<= XY_SHIFT - shift;
    p0.y <<= XY_SHIFT - shift;

    assert(0 <= shift && shift <= XY_SHIFT);
    xmin = xmax = v[0].x;
    ymin = ymax = v[0].y;
    for(i = 0; i < npts; ++i) {
        Point p = v[i];
        if(p.y < ymin) {
            ymin = p.y;
            imin = i;
        }
        ymax = _max(ymax, p.y);
        xmax = _max(xmax, p.x);
        xmin = _min(xmin, p.x);
        p.x <<= XY_SHIFT - shift;
        p.y <<= XY_SHIFT - shift;
        Line2( img, p0, p, color );
        p0 = p;
    }
    xmin = (xmin + delta) >> shift;
    xmax = (xmax + delta) >> shift;
    ymin = (ymin + delta) >> shift;
    ymax = (ymax + delta) >> shift;
    if( npts < 3 || (int)xmax < 0 || (int)ymax < 0 || (int)xmin >= width || (int)ymin >= height )
        return;
    ymax = _min( ymax, height - 1 );
    edge[0].idx = edge[1].idx = imin;

    edge[0].ye = edge[1].ye = y = (int)ymin;
    edge[0].di = 1;
    edge[1].di = npts - 1;

    edge[0].x = edge[1].x = -XY_ONE;
    edge[0].dx = edge[1].dx = 0;

    ptr += img->w_stride*y;
    do {
        if( line_type < LINE_AA || y < (int)ymax || y == (int)ymin ) {
            for( i = 0; i < 2; i++ ) {
                if( y >= edge[i].ye ) {
                    int idx0 = edge[i].idx, di = edge[i].di;
                    int idx = idx0 + di;
                    if (idx >= npts) idx -= npts;
                    int ty = 0;

                    for (; edges-- > 0; ) {
                        ty = (int)((v[idx].y + delta) >> shift);
                        if (ty > y) {
                            int64_t xs = v[idx0].x;
                            int64_t xe = v[idx].x;
                            if (shift != XY_SHIFT) {
                                xs <<= XY_SHIFT - shift;
                                xe <<= XY_SHIFT - shift;
                            }

                            edge[i].ye = ty;
                            edge[i].dx = ((xe - xs)*2 + (ty - y)) / (2 * (ty - y));
                            edge[i].x = xs;
                            edge[i].idx = idx;
                            break;
                        }
                        idx0 = idx;
                        idx += di;
                        if (idx >= npts) idx -= npts;
                    }
                }
            }
        }

        if (edges < 0)
            break;

        if (y >= 0) {
            int left = 0, right = 1;
            if (edge[0].x > edge[1].x) {
                left = 1, right = 0;
            }

            int xx1 = (int)((edge[left].x + delta1) >> XY_SHIFT);
            int xx2 = (int)((edge[right].x + delta2) >> XY_SHIFT);

            if( xx2 >= 0 && xx1 < width ) {
                if( xx1 < 0 )
                    xx1 = 0;
                if( xx2 >= width )
                    xx2 = width - 1;
                ICV_HLINE( ptr, xx1, xx2, color, img->c );
            }
        } else {
            // TODO optimize scan for negative y
        }

        edge[0].x += edge[0].dx;
        edge[1].x += edge[1].dx;
        ptr += (img->w_stride);
    } while( ++y <= (int)ymax );
}

static void ThickLine(Img* img, Point p0, Point p1,
                      const void* color, int thickness,
                      int line_type, int flags, int shift) {
    static const float INV_XY_ONE = 1./XY_ONE;
    p0.x <<= XY_SHIFT - shift;
    p0.y <<= XY_SHIFT - shift;
    p1.x <<= XY_SHIFT - shift;
    p1.y <<= XY_SHIFT - shift;
    if( thickness <= 1 ) {
        Line2(img, p0, p1, color);
    }else {
        Point pt[4], dp={0,0};
        float dx = (p0.x - p1.x)*INV_XY_ONE, dy = (p1.y - p0.y)*INV_XY_ONE;
        float r = dx * dx + dy * dy;
        int oddThickness = thickness & 1;
        thickness <<= XY_SHIFT - 1;

        if( fabs(r) > DBL_EPSILON ) {
            r = (thickness + oddThickness*XY_ONE*0.5)/sqrt(r);
            dp.x = roundf( dy * r );
            dp.y = roundf( dx * r );

            pt[0].x = p0.x + dp.x;
            pt[0].y = p0.y + dp.y;
            pt[1].x = p0.x - dp.x;
            pt[1].y = p0.y - dp.y;
            pt[2].x = p1.x - dp.x;
            pt[2].y = p1.y - dp.y;
            pt[3].x = p1.x + dp.x;
            pt[3].y = p1.y + dp.y;
            FillConvexPoly( img, pt, 4, color, line_type, XY_SHIFT );
        }
    }
}

static void PolyLine(Img* img, const Point* v, int count,
                     int is_closed, const void* color,
                     int thickness, int line_type, int shift) {
    if(!v || count <= 0)
        return;

    int i = is_closed ? count - 1 : 0;
    int flags = 2 + !is_closed;
    Point p0;
    assert(0 <= shift && shift <= XY_SHIFT && thickness >= 0);
    p0 = v[i];
    for(i = !is_closed; i < count; ++i) {
        Point p = v[i];
        ThickLine(img, p0, p, color, thickness, line_type, flags, shift);
        p0 = p;
        flags = 2;
    }
}

void sample_draw_box_for_image(Img* img, Point pt1, Point pt2, const void* color, int thickness) {
    if(!img) {
        fprintf(stderr, "[Error] : NULL IMAGE, %s, %d\n", __func__, __LINE__);
        exit(1);
    } else if(img->data == NULL) {
        fprintf(stderr, "[Error] : NULL IMAGE DATA, %s, %d\n", __func__, __LINE__);
        exit(1);
    } else if(!color) {
        fprintf(stderr, "[Error] : NULL COLOR, %s, %d\n", __func__, __LINE__);
        exit(1);
    }
    int lineType = 8;//change me
    Point pt[4];
    int shift = 0;
    pt[0] = pt1;

    pt[1].x = pt2.x;
    pt[1].y = pt1.y;

    pt[2] = pt2;

    pt[3].x = pt1.x;
    pt[3].y = pt2.y;

    if( thickness >= 0 )
        PolyLine( img, pt, 4, 1, color, thickness, lineType, shift );
    else
        FillConvexPoly( img, pt, 4, color, lineType, shift );
}

void sample_draw_line_for_image(Img* img, Point pt1, Point pt2, const void* color, int thickness) {
    if(!img) {
        fprintf(stderr, "[Error] : NULL IMAGE, %s, %d\n", __func__, __LINE__);
        exit(1);
    } else if(img->data == NULL) {
        fprintf(stderr, "[Error] : NULL IMAGE DATA, %s, %d\n", __func__, __LINE__);
        exit(1);
    } else if(!color) {
        fprintf(stderr, "[Error] : NULL COLOR, %s, %d\n", __func__, __LINE__);
        exit(1);
    }
    int lineType = 8;
    ThickLine(img, pt1, pt2, color, thickness, lineType, 3, 0 );
}
