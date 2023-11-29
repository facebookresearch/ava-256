#include "utils.h"
#include <cassert>

void utils::memcpy_bytes(const py::object& obj, void* __restrict__ dst, int expected_size) {
  PyObject* _obj = obj.ptr();
  char* buffer = nullptr;
  Py_ssize_t length = 0;
  if (PyBytes_AsStringAndSize(_obj, &buffer, &length) != 0) {
    throw py::error_already_set();
  }
  if (expected_size != length) {
    throw std::invalid_argument("Size of the bytes object does not match the expected value");
  }
  assert(buffer);
  memcpy(dst, buffer, length);
}

bool utils::memcpy_bytes_no_throw(
    const py::object& obj,
    void* __restrict__ dst,
    int expected_size) {
  PyObject* _obj = obj.ptr();
  char* buffer = nullptr;
  Py_ssize_t length = 0;
  if (PyBytes_AsStringAndSize(_obj, &buffer, &length) != 0) {
    return false;
  }
  if (expected_size != length) {
    return false;
  }
  assert(buffer);
  memcpy(dst, buffer, length);
  return true;
}

// see https://github.com/pybind/pybind11/issues/1236#issuecomment-527730864
py::object utils::make_bytes(size_t size, char** ptr) {
  auto bytesObject = (PyBytesObject*)PyObject_Malloc(offsetof(PyBytesObject, ob_sval) + size + 1);
  PyObject_INIT_VAR(bytesObject, &PyBytes_Type, size);
  bytesObject->ob_shash = -1;
  bytesObject->ob_sval[size] = '\0';
  *ptr = bytesObject->ob_sval;
  return py::reinterpret_steal<py::object>((PyObject*)bytesObject);
};

void utils::humanize(int64_t& x, int& d, const char** s) {
  const char* suf[] = {"", "k", "M", "G"};
  int i = 0;
  d = 0;
  for (; i < 4 && x > 10000; ++i) {
    d = x % 1024;
    d = 10 * (d + 512) / 1024;
    x /= 1024;
  }
  if (d == 10) {
    ++x;
    d = 0;
  }
  *s = suf[i];
}
