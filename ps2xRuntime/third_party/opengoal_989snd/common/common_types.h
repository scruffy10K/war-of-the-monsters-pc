// Shim for the OpenGOAL "common/common_types.h" used by the vendored 989snd engine.
#pragma once
#include <cstddef>
#include <cstdint>
using u8 = uint8_t;
using s8 = int8_t;
using u16 = uint16_t;
using s16 = int16_t;
using u32 = uint32_t;
using s32 = int32_t;
using u64 = uint64_t;
using s64 = int64_t;
// Standard headers OpenGOAL's own common headers bring in transitively.
#include <algorithm>
#include <array>
#include <cstring>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>
