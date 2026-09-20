#include "hit.h"
#include <cmath>

// Тестовая фигура — piriform при a = 2.
// Поверхность задаётся уравнением x^4 - a*x^3 + a^2*(y^2 + z^2) = 0
//равносильно a^2*(y^2 + z^2) = x^3*(a - x)
// Тогда условие принадлежности точки внутренности фигуры: y^2 + z^2 <= x^3*(a - x) / a^2,   при 0 <= x <= a
static const float A = 2.0f;

bool hit_test(float x, float y, float z) {
    // У piriform область существует только при 0 <= x <= A, потому что выражение x^3*(A - x) должно быть неотрицательным.
    if (x < 0.0f || x > A) {
        return false;
    }

    // Квадрат расстояния от точки до оси Ox в плоскости yz.
    const float r2 = y * y + z * z;

    // Правая часть условия принадлежности piriform:
    // y^2 + z^2 <= x^3*(A - x)/A^2
    const float rhs = (x * x * x * (A - x)) / (A * A);

    // Если квадрат радиуса точки не превышает допустимый, то точка лежит внутри фигуры или на её границе.
    return r2 <= rhs;
}

const float* get_axis_range() {
    // Границы параллелепипеда, внутри которого генерируются случайные точки:
    // [xmin, xmax, ymin, ymax, zmin, zmax]
    // По x:
    // фигура лежит в диапазоне [0, A].
    // По y и z:
    // максимальный радиус берётся из функции r^2(x) = x^3*(A - x)/A^2
    // максимум достигается при x = 3A/4,поэтому r_max = 3*sqrt(3)*A / 16.
    static float range[6];

    const float r_max = 3.0f * std::sqrt(3.0f) * A / 16.0f;

    range[0] = 0.0f;
    range[1] = A;
    range[2] = -r_max;
    range[3] =  r_max;
    range[4] = -r_max;
    range[5] =  r_max;

    return range;
}