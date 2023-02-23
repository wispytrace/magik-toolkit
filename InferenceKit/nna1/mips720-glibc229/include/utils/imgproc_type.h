#ifndef __MAGIK_INFERENCEKIT_VENUS_INCLUDE_UTILS_IMGPROC_TYPE_H__
#define __MAGIK_INFERENCEKIT_VENUS_INCLUDE_UTILS_IMGPROC_TYPE_H__
namespace magik {
namespace venus {

enum class PaddingType : int {
    /*no padding*/
    NONE = -1,
    /*
     * (1). BGR0, eg: src_shape=[1, 3, 5, 4], dst_shape=[1, 5, 7, 4]:
     * padval = 0;
     * +--------------------+       +----------------------------+
     * |BGR0BGR0BGR0BGR0BGR0|       |BGR0BGR0BGR0BGR0BGR000000000|
     * |BGR0BGR0BGR0BGR0BGR0| ====> |BGR0BGR0BGR0BGR0BGR000000000|
     * |BGR0BGR0BGR0BGR0BGR0|       |BGR0BGR0BGR0BGR0BGR000000000|
     * +--------------------+       |0000000000000000000000000000|
     *                              |0000000000000000000000000000|
     *                              +----------------------------+
     *
     * (2). NV12, eg: src_shape=[1, 4, 6, 1], dst_shape=[1, 6, 8, 1]:
     * padval_y = 16;
     * padval_uv = 128;
     * +------+                     +-------------------------------+
     * |YYYYYY|                     |Y   Y   Y   Y   Y   Y   16  16 |
     * |YYYYYY|                     |Y   Y   Y   Y   Y   Y   16  16 |
     * |YYYYYY|       =====>        |Y   Y   Y   Y   Y   Y   16  16 |
     * |YYYYYY|                     |Y   Y   Y   Y   Y   Y   16  16 |
     * |UVUVUV|                     |16  16  16  16  16  16  16  16 |
     * |UVUVUV|                     |16  16  16  16  16  16  16  16 |
     * +------+                     |U   V   U   V   U   V   128 128|
     *                              |U   V   U   V   U   V   128 128|
     *                              |128 128 128 128 128 128 128 128|
     *                              +-------------------------------+
     */
    BOTTOM_RIGHT = 0,

    /*
     * (1). BGR0, eg: src_shape=[1, 3, 5, 4], dst_shape=[1, 5, 7, 4]:
     * padval = 0;
     * +--------------------+        +----------------------------+
     * |BGR0BGR0BGR0BGR0BGR0|        |0000000000000000000000000000|
     * |BGR0BGR0BGR0BGR0BGR0| =====> |0000BGR0BGR0BGR0BGR0BGR00000|
     * |BGR0BGR0BGR0BGR0BGR0|        |0000BGR0BGR0BGR0BGR0BGR00000|
     * +--------------------+        |0000BGR0BGR0BGR0BGR0BGR00000|
     *                               |0000000000000000000000000000|
     *                               +----------------------------+
     *
     * (2). NV12, eg: src_shape=[1, 4, 6, 1], dst_shape=[1, 8, 10, 1]:
     * padval_y = 16;
     * padval_uv = 128;
     * +------+                     +---------------------------------------+
     * |YYYYYY|                     |16  16  16  16  16  16  16  16  16  16 |
     * |YYYYYY|                     |16  16  16  16  16  16  16  16  16  16 |
     * |YYYYYY|       =====>        |16  16  Y   Y   Y   Y   Y   Y   16  16 |
     * |YYYYYY|                     |16  16  Y   Y   Y   Y   Y   Y   16  16 |
     * |UVUVUV|                     |16  16  Y   Y   Y   Y   Y   Y   16  16 |
     * |UVUVUV|                     |16  16  Y   Y   Y   Y   Y   Y   16  16 |
     * +------+                     |16  16  16  16  16  16  16  16  16  16 |
     *                              |16  16  16  16  16  16  16  16  16  16 |
     *                              |128 128 128 128 128 128 128 128 128 128|
     *                              |128 128 U   V   U   V   U   V   128 128|
     *                              |128 128 U   V   U   V   U   V   128 128|
     *                              |128 128 128 128 128 128 128 128 128 128|
     *                              +---------------------------------------+
     */
    SYMMETRY = 1
};
enum class AddressLocate : int {
    NMEM_VIRTUAL = 0,  // virtual address in nmem
    RMEM_PHYSICAL = 1, // physical address in rmem
};
} // namespace venus
} // namespace magik
#endif /* __MAGIK_INFERENCEKIT_VENUS_INCLUDE_UTILS_IMGPROC_TYPE_H__ */
