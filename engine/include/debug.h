#include <iostream>
#include <vector>
#include <pybind11/numpy.h>

inline void err(std::string_view message) {
  std::cerr << "\033[31m[!] " << message << "\033[0m\n";            // Red
}

inline void warn(std::string_view message) {
  std::cout << "\033[33m[-] " << message << "\033[0m" << std::endl; // Yellow
}

inline void pass(std::string_view message) {
  std::cout << "\033[32m[+] " << message << "\033[0m" << std::endl; // Green
}

inline void log(std::string_view message) {
  std::cout << "[*] " << message << std::endl;                      // White (default/normal)
}

// Specifically to print matrices
template <typename T>
void print_mat(const std::vector<std::vector<T>>& mat, std::ostream& os = std::cout) {
  os << "[\n";
  for (size_t i = 0; i < mat.size(); ++i) {
    os << "  [";
    for (size_t j = 0; j < mat[i].size(); ++j) {
      os << mat[i][j];
      if (j + 1 < mat[i].size()) os << ", ";
    }
    os << "]";
    if (i + 1 < mat.size()) os << ",";
    os << "\n";
  }
  os << "]" << std::endl;
}

// Use to print out arrays in numpy style
inline pybind11::array_t<float> vec_to_numpy(const std::vector<std::vector<float>>& mat) {
  if (mat.empty()) return pybind11::array_t<float>(std::vector<size_t>{0, 0});
  size_t rows = mat.size();
  size_t cols = mat[0].size();
  pybind11::array_t<float> arr({rows, cols});
  auto buf = arr.mutable_data();

  for (size_t i = 0; i < rows; ++i) {
    if (mat[i].size() != cols) 
      throw std::runtime_error("All rows must have same length");
    for (size_t j = 0; j < cols; ++j) {
      buf[i*cols + j] = mat[i][j];
    }
  }
  return arr;
}

