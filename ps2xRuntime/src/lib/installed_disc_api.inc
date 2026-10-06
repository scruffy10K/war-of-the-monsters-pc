// Private host I/O interface; guest ABI and generated-code headers are unchanged.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
namespace wotm_disc {
bool mount(const std::filesystem::path& index);
uint64_t mountedSize();
bool readMounted(uint64_t offset, void* data, size_t length, size_t& got);
bool location(const std::filesystem::path& asset, uint32_t& lbn);
std::FILE* open(const std::filesystem::path& path, uint64_t* length = nullptr);
bool read(std::FILE* file, uint64_t offset, void* data, size_t length, size_t& got);
void close(std::FILE* file);
}
