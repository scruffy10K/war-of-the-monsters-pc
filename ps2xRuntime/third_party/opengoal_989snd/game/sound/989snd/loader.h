// Shim replacing OpenGOAL's loader.h: this project loads War of the Monsters
// SBlk banks itself (ps2xRuntime/src/lib/ps2_snd989_sfx.cpp).
#pragma once
namespace snd
{
class SoundBank;
using BankHandle = SoundBank *;
} // namespace snd
