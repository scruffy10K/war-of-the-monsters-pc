// Shim for the OpenGOAL logger: messages are dropped.
#pragma once
namespace lg
{
template <typename... Args> inline void error(Args &&...) {}
template <typename... Args> inline void warn(Args &&...) {}
template <typename... Args> inline void info(Args &&...) {}
template <typename... Args> inline void debug(Args &&...) {}
template <typename... Args> inline void print(Args &&...) {}
} // namespace lg
