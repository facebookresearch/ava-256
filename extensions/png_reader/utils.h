#pragma once
#include <pybind11/pybind11.h>
#include <cinttypes>

namespace py = pybind11;

namespace utils {
void memcpy_bytes(const py::object& obj, void* dst, int expected_size);

bool memcpy_bytes_no_throw(const py::object& obj, void* dst, int expected_size);

py::object make_bytes(size_t size, char** ptr);

void humanize(int64_t& x, int& d, const char** s);

template <typename T>
const char* type2str();

#define REG_INT_TYPE(X)                  \
  template <>                            \
  inline const char* type2str<X##_t>() { \
    return #X;                           \
  }
REG_INT_TYPE(uint8)
REG_INT_TYPE(uint16)
REG_INT_TYPE(uint32)
REG_INT_TYPE(uint64)
REG_INT_TYPE(int8)
REG_INT_TYPE(int16)
REG_INT_TYPE(int32)
REG_INT_TYPE(int64)

#undef REG_INT_TYPE
} // namespace utils
