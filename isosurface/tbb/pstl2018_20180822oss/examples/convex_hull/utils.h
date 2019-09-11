/*
    Copyright (c) 2017-2018 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.




*/

#include <random>
#include <string>
#include <iostream>

namespace util {

    template <typename T>
    struct point {
        T x;
        T y;
        point() {}
        point(T xx, T yy) : x(xx), y(yy) {}
        point(const point& p) : x(p.x), y(p.y) {}

        bool operator ==(const point &p2) const {
            return (this->x == p2.x && this->y == p2.y);
        }
        bool operator !=(const point &p2) const {
            return !(*(this) == p2);
        }
        bool operator < (const point & p2) const {
            return (this->x == p2.x ? this->y < p2.y : this->x < p2.x);
        }
    };

    template <typename T>
    T cross_product(const point<T>& start, const point<T>& end1, const point<T>& end2) {
        return ((end1.x - start.x)*(end2.y - start.y) - (end2.x - start.x)*(end1.y - start.y));
    }

    template <typename T>
    std::ostream& operator <<(std::ostream& ostr, point<T> _p) {
        return ostr << _p.x << ',' << _p.y;
    }

    template <typename T>
    std::istream& operator >>(std::istream& istr, point<T> _p) {
        return istr >> _p.x >> _p.y;
    }

    // The variable is declared out of the scope of random_point() to avoid code generation issues with some compilers
    thread_local static std::default_random_engine rd;

    template <typename T>
    point<T> random_point() {
        const int rand_max = 10000;
        std::uniform_int_distribution<int> dist(-rand_max, rand_max);
        T x = dist(rd);
        T y = dist(rd);
        const double r = x*x + y*y;
        if (r > rand_max) {
            x /= r;
            y /= r;
        }
        return point<T>(x, y);
    }

}
