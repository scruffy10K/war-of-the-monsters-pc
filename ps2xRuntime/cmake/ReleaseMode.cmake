include(CheckIPOSupported)

check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)

# Link-time optimisation re-optimises the whole executable on every link, so a
# one-file change costs minutes; allow turning it off for iteration builds.
option(PS2X_ENABLE_LTO "Whole-program / link-time optimisation in Release" ON)
# /fp:fast lets the compiler reorder and contract float maths. VU0/VU1 results
# must follow the PS2's exact operation order (clamping, flags, Q timing), so
# it stays off unless explicitly requested.
option(PS2X_ENABLE_FAST_MATH "Use /fp:fast in Release (changes float results)" OFF)

function(EnableFastReleaseMode TargetName)
    message("> Enabling optimization for: ${TargetName}")
    if(MSVC)
        target_compile_options(${TargetName} PRIVATE
            $<$<CONFIG:Release>:
                /O2 # speed
                /Ob2 # inline aggressively
                /Oi # intrinsics
                $<$<BOOL:${PS2X_ENABLE_LTO}>:/GL> # whole program opt
                /Gy # function-level linking
                /Gw # global data in COMDAT
                /GF # string pooling
                /Zc:inline # remove unreferenced inline
                $<$<BOOL:${PS2X_ENABLE_FAST_MATH}>:/fp:fast> # fast math (changes VU float results)
                /DNDEBUG
                /arch:AVX2 # Advanced Vector Extensions 2
                /GS- # Disable Buffer Security Check (faster)
                $<$<NOT:$<CXX_COMPILER_ID:Clang>>:/Qspectre-> # Disable Spectre mitigations (MSVC only; clang-cl has no such flag)
                # clang may fuse a*b+c into one FMA instruction under /arch:AVX2,
                # which rounds differently from the PS2's separate multiply and add.
                $<$<AND:$<CXX_COMPILER_ID:Clang>,$<NOT:$<BOOL:${PS2X_ENABLE_FAST_MATH}>>>:/clang:-ffp-contract=off>
            >
        )

        if(TARGET ${TargetName})
            target_link_options(${TargetName} PRIVATE
                $<$<CONFIG:Release>:
                    $<$<BOOL:${PS2X_ENABLE_LTO}>:/LTCG> # link-time code generation
                    /OPT:REF # remove unreferenced
                    /OPT:ICF # fold identical COMDATs
                >
            )
        endif()
    endif()

    if(IPO_SUPPORTED AND PS2X_ENABLE_LTO)
        set_property(TARGET ${TargetName} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    elseif(PS2X_ENABLE_LTO)
        message(WARNING "Interprocedural optimization not supported: ${ipo_error}")
    endif()
endfunction()