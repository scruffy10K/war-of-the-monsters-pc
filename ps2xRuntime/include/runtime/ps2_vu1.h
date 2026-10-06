#ifndef PS2_VU1_H
#define PS2_VU1_H

#include <array>
#include <cstdint>
#include <vector>

class GS;
class PS2Memory;

struct VU1State
{
    float vf[32][4];
    int32_t vi[16];
    float acc[4];
    float q;
    float p;
    float i;
    uint32_t r;
    uint32_t pc;
    uint32_t mac;
    uint32_t clip;
    uint32_t status;
    uint64_t cycles;
    bool ebit;
    bool haltAfterDelaySlot;
    bool dBitEnabled;
    bool tBitEnabled;
    bool stoppedByD;
    bool stoppedByT;
    uint32_t top;  // VIF TOP visible to XTOP
    uint32_t itop; // VIF ITOP visible to XITOP

    bool branchPending;
    uint32_t branchTarget;
    uint32_t branchDelay;
};

class VU1Interpreter
{
public:
    enum class Unit : uint8_t
    {
        VU0,
        VU1
    };

    explicit VU1Interpreter(Unit unit = Unit::VU1);

    void reset();

    void execute(uint8_t *vuCode, uint32_t codeSize,
                 uint8_t *vuData, uint32_t dataSize,
                 GS &gs, PS2Memory *memory = nullptr,
                 uint32_t startPC = 0, uint32_t top = 0, uint32_t itop = 0,
                 uint32_t maxCycles = 65536);

    void resume(uint8_t *vuCode, uint32_t codeSize,
                uint8_t *vuData, uint32_t dataSize,
                GS &gs, PS2Memory *memory = nullptr,
                uint32_t top = 0, uint32_t itop = 0, uint32_t maxCycles = 65536);

    VU1State &state() { return m_state; }
    const VU1State &state() const { return m_state; }

    // Specialised VU1 program execution (ps2_vu1_native.cpp) reaches the
    // interpreter's pipeline state through this, so adding or changing native
    // code never needs another edit of this widely included header.
    friend struct Vu1NativeAccess;

private:
    enum Pipeline : uint8_t
    {
        PipelineNone = 0,
        PipelineFmac,
        PipelineLsu,
        PipelineFdiv,
        PipelineEfu,
        PipelineIalu,
        PipelineBranch,
        PipelineXgkick
    };

    struct VfAccess
    {
        uint8_t reg = 0;
        uint8_t lanes = 0;
    };

    struct InstructionUsage
    {
        std::array<VfAccess, 2> vfRead{};
        VfAccess vfWrite{};
        uint8_t vfReadCount = 0;
        uint16_t viRead = 0;
        uint16_t viWrite = 0;
        uint8_t accRead = 0;
        uint8_t accWrite = 0;
        uint8_t latency = 0;
        uint8_t vfLatency = 0;
        uint8_t viLatency = 0;
        Pipeline pipeline = PipelineNone;
        bool waitQ = false;
        bool waitP = false;
        bool readsClip = false;
        bool writesClip = false;
        bool delaysNextBranchRead = false;
        bool reserved = false;
    };

    struct DecodedInstructionPair
    {
        uint32_t lower = 0;
        uint32_t upper = 0;
        InstructionUsage lowerUsage{};
        InstructionUsage upperUsage{};
        bool iBit = false;
        bool eBit = false;
        bool mBit = false;
        bool dBit = false;
        bool tBit = false;
        uint8_t upperVfShadowReg = 0;
        uint8_t suppressedLowerVf = 0;
        // Union of both halves' register reads, so calculatePairReadyCycle can
        // test "does this pair read anything still in flight" with two ANDs
        // instead of walking every declared read on every instruction.
        uint32_t vfReadMask = 0;
        uint16_t viReadMask = 0;
        bool readsAcc = false;
    };

    struct FlagPipelineEntry
    {
        uint64_t readyCycle = 0;
        uint64_t issueCycle = 0;
        uint32_t mac = 0;
        uint32_t status = 0;
        uint32_t extraSticky = 0;
        uint32_t clip = 0;
        bool valid = false;
        bool writesMac = false;
        bool writesStatus = false;
        bool writesSticky = false;
        bool writesClip = false;
        // PS2X_VU1_FLAG_CHECK: eager shadow already applied this entry.
        bool shadowApplied = false;
    };

    struct ScalarPipelineEntry
    {
        uint64_t readyCycle = 0;
        float value = 0.0f;
        uint32_t statusDi = 0;
        bool valid = false;
    };

    struct PendingStore
    {
        uint64_t readyCycle = 0;
        uint32_t address = 0;
        std::array<uint32_t, 4> words{};
        uint8_t laneMask = 0;
        bool valid = false;
    };

    struct PendingVfWrite
    {
        uint64_t readyCycle = 0;
        uint64_t sequence = 0;
        std::array<float, 4> value{};
        uint8_t reg = 0;
        uint8_t laneMask = 0;
        bool valid = false;
    };

    struct PendingViWrite
    {
        uint64_t readyCycle = 0;
        uint64_t sequence = 0;
        int32_t value = 0;
        uint8_t reg = 0;
        bool valid = false;
    };

    struct PendingAccWrite
    {
        uint64_t readyCycle = 0;
        uint64_t sequence = 0;
        std::array<float, 4> value{};
        uint8_t laneMask = 0;
        bool valid = false;
    };

    struct XgkickPipeline
    {
        static constexpr uint32_t kBufferSize = 0x10000u;
        std::array<uint8_t, kBufferSize> packet{};
        uint32_t sourceAddress = 0;
        uint32_t totalBytes = 0;
        uint32_t copiedBytes = 0;
        uint32_t currentTagEnd = 0;
        uint32_t cycleCredit = 0;
        uint64_t issueCycle = 0;
        bool active = false;
        bool currentTagEop = false;
    };

    static constexpr uint32_t kFmacLatency = 4u;
    static constexpr uint32_t kAccForwardLatency = 1u;
    // Flag entries retire lazily (retireFlags) at the instructions that read
    // MAC/status/clip, not every cycle, so more can be in flight at once.
    static constexpr uint32_t kMaxFlagEntries = 32u;
    static constexpr uint32_t kMaxPendingStores = 8u;
    static constexpr uint32_t kMaxPendingVfWrites = 16u;
    static constexpr uint32_t kMaxPendingViWrites = 8u;
    static constexpr uint32_t kMaxPendingAccWrites = 8u;
    static constexpr uint32_t kMaxDecodedPairs = 0x4000u / 8u;

    Unit m_unit;
    VU1State m_state;
    std::array<DecodedInstructionPair, kMaxDecodedPairs> m_decodedCodeCache{};
    const uint8_t *m_cachedVuCode = nullptr;
    const PS2Memory *m_cachedMemory = nullptr;
    uint32_t m_cachedCodeSize = 0;
    uint64_t m_cachedCodeGeneration = 0;
    bool m_decodedCodeCacheValid = false;

    std::array<FlagPipelineEntry, kMaxFlagEntries> m_flagPipeline{};
    ScalarPipelineEntry m_fdiv{};
    std::array<ScalarPipelineEntry, 2> m_efu{};
    std::array<PendingStore, kMaxPendingStores> m_storePipeline{};
    std::array<PendingVfWrite, kMaxPendingVfWrites> m_vfWritePipeline{};
    std::array<PendingViWrite, kMaxPendingViWrites> m_viWritePipeline{};
    std::array<PendingAccWrite, kMaxPendingAccWrites> m_accWritePipeline{};
    XgkickPipeline m_xgkick{};
    std::array<std::array<uint64_t, 4>, 32> m_vfReady{};
    std::array<uint64_t, 16> m_viReady{};
    std::array<uint64_t, 4> m_accReady{};
    std::array<std::array<uint64_t, 4>, 32> m_vfLatestWrite{};
    std::array<uint64_t, 16> m_viLatestWrite{};
    std::array<uint64_t, 4> m_accLatestWrite{};

    uint64_t m_cycle = 0;
    // Fast-path watermarks. The pipeline arrays above total ~51 slots and were
    // scanned in full on every emulated cycle, and the m_*Ready arrays another
    // ~50 on every instruction pair, even though both are almost always empty --
    // together that was the bulk of the ~259 ns/instruction the VU1 interpreter
    // was costing. These two values let the scans exit early:
    //   m_pipelineNextReady - smallest readyCycle among *valid* pipeline entries
    //                         (max() when nothing is in flight).
    //   m_maxReadyCycle     - largest value stored in m_vfReady/m_viReady/
    //                         m_accReady, i.e. the last cycle any write lands.
    // Both are only ever wrong in the safe direction: a too-small
    // m_pipelineNextReady or a too-large m_maxReadyCycle just costs a scan that
    // finds nothing. Never let them drift the other way -- every site that makes
    // a pipeline entry valid must call notePipelineReady().
    uint64_t m_pipelineNextReady = ~0ull;
    uint64_t m_maxReadyCycle = 0;
    // Pending-write masks: bit set <=> that register MAY have a lane whose
    // m_*Ready is still > m_cycle. Set by markPairWrites, cleared lazily in
    // calculatePairReadyCycle once the register's last write has landed (safe:
    // m_cycle only grows). Only ever wrong in the safe direction (a stale set
    // bit costs a scan that finds nothing). mutable: the scan is const.
    mutable uint32_t m_vfPendingMask = 0;
    mutable uint16_t m_viPendingMask = 0;
    mutable bool m_accPending = false;
    // Occupancy masks: bit i set <=> the pipeline array's slot i is .valid.
    // Commit walks only live slots and inserts find a free slot with one
    // bit-scan, instead of both scanning every fixed slot on every cycle.
    // Diag builds verify the invariant at the end of each run ([vu1:mask]).
    uint32_t m_flagMask = 0;
    uint32_t m_storeMask = 0;
    uint32_t m_vfWriteMask = 0;
    uint32_t m_viWriteMask = 0;
    uint32_t m_accWriteMask = 0;
    // Stage 1a: leave VF/VI/ACC results where exec wrote them (see run()).
    bool m_immediateWrites = false;
    // Completed XGKICK packets are appended here when set; with
    // m_gifCaptureOnly they are captured instead of sent to the GS.
    std::vector<uint8_t> *m_gifCapture = nullptr;
    bool m_gifCaptureOnly = false;
    uint64_t m_nextWriteSequence = 0;
    uint64_t m_efuResourceReady = 0;
    uint32_t m_workingClip = 0;
    // PS2X_VU1_FLAG_CHECK=1: MAC/status/clip maintained the old way (applied
    // the cycle each entry becomes ready) for comparison with lazy retirement.
    uint32_t m_shadowMac = 0;
    uint32_t m_shadowStatus = 0;
    uint32_t m_shadowClip = 0;
    // Flag entries in issue order. Every entry has the same latency, so issue
    // order is ready order and retireFlags pops from the head with no search.
    std::array<uint8_t, kMaxFlagEntries> m_flagOrder{};
    uint32_t m_flagOrderHead = 0;
    uint32_t m_flagOrderCount = 0;
    // Native immediate flag updates (ps2_vu1_native.h): the cycle at which the
    // last update applied early would have retired. pipelinesPending() keeps
    // reporting pending until then so the program-end drain is unchanged.
    uint64_t m_flagTailReady = 0;
    uint32_t m_currentUpperInstruction = 0;
    int32_t m_viBranchBackupValue = 0;
    uint8_t m_viBranchBackupReg = 0;
    bool m_viBranchBackupValid = false;
    uint8_t *m_activeVuData = nullptr;
    uint32_t m_activeVuDataSize = 0;
    GS *m_activeGs = nullptr;
    PS2Memory *m_activeMemory = nullptr;
    bool m_stopRequested = false;
    bool m_pendingHaltD = false;
    bool m_pendingHaltT = false;

    void run(uint8_t *vuCode, uint32_t codeSize,
             uint8_t *vuData, uint32_t dataSize,
             GS &gs, PS2Memory *memory, uint32_t maxCycles);

    static constexpr InstructionUsage decodeUpperUsage(uint32_t upper, Unit unit);
    static constexpr InstructionUsage decodeLowerUsage(uint32_t lower, Unit unit);
    static constexpr void addVfRead(InstructionUsage &usage, uint8_t reg, uint8_t lanes);
    static constexpr void addVfWrite(InstructionUsage &usage, uint8_t reg, uint8_t lanes);
    static constexpr uint8_t vfReadLanes(const InstructionUsage &usage, uint8_t reg);
    DecodedInstructionPair decodeInstructionPair(const uint8_t *vuCode, uint32_t pc) const;
    // Pure decode of one pair from its two words. constexpr so the VU1 native
    // programs (ps2_vu1_native.h) fold the whole usage/latency/hazard picture
    // of each instruction at compile time using this exact decoder.
    static constexpr DecodedInstructionPair decodePairWords(uint32_t upper, uint32_t lower, Unit unit);
    DecodedInstructionPair getDecodedInstructionPairForPc(const uint8_t *vuCode, uint32_t codeSize, PS2Memory *memory, uint32_t pc);
    void rebuildDecodedCodeCache(const uint8_t *vuCode, uint32_t codeSize, const PS2Memory *memory, uint64_t generation);

    void execUpper(uint32_t instr);
    void execLower(uint32_t instr, uint8_t *vuData, uint32_t dataSize, GS &gs, PS2Memory *memory, uint32_t upperInstr);

    void applyDest(float *dst, const float *result, uint8_t dest);
    void applyDestAcc(const float *result, uint8_t dest);
    void applyFmacDest(float *dst, float *result, uint8_t dest);
    void applyFmacDestAcc(float *result, uint8_t dest);
    void normalizeFmacResult(float *result, uint8_t dest, uint8_t laneFlags[4]);
    bool calculateFmacExactResult(uint32_t component, long double &result) const;
    uint8_t normalizeFmacExactResult(float &value, long double exactResult) const;
    uint32_t calculateFmacProductSticky(uint8_t dest) const;
    void updateFmacFlags(const uint8_t laneFlags[4], uint8_t dest, uint32_t extraSticky);
    void queueFsset(uint16_t immediate);
    void queueClip(uint32_t clip);
    void queueFcset(uint32_t clip);
    void queueQ(float value, uint32_t latency, uint32_t statusDi);
    void queueP(float value, uint32_t latency);
    void queueStore(uint32_t address, const uint32_t words[4], uint8_t laneMask);
    // Must be called wherever a pipeline entry becomes valid; see the comment on
    // m_pipelineNextReady.
    void notePipelineReady(uint64_t readyCycle)
    {
        if (readyCycle < m_pipelineNextReady)
            m_pipelineNextReady = readyCycle;
    }

    void queueVfWrite(uint8_t reg, uint8_t laneMask, const float value[4], uint32_t latency);
    void queueViWrite(uint8_t reg, int32_t value, uint32_t latency);
    void queueAccWrite(uint8_t laneMask, const float value[4], uint32_t latency);
    void startXgkick(uint32_t qwordAddress);

    void resetScheduler();
    void commitReadyPipelines();
    // Apply every flag-pipeline entry whose readyCycle <= m_cycle, in issue
    // order, then free it. Called on demand: by the flag-reading instructions,
    // when the pool is full, at program end, and before state is published.
    void retireFlags();
    void pushFlagOrder(uint32_t slot);
    // retireFlags() + (PS2X_VU1_FLAG_CHECK) compare against the eager shadow.
    void syncFlagsForRead(const char *op);
    void advanceOneCycle();
    void advanceTo(uint64_t targetCycle);
    void flushPipelines();
    void progressXgkick();
    void finishXgkick();
    uint64_t calculatePairReadyCycle(const DecodedInstructionPair &decoded) const;
    void markPairWrites(const DecodedInstructionPair &decoded);

    // Stage 1 VU1 fast path + differential harness (see ps2_vu1_core.cpp).
    void diffBegin(uint8_t *vuCode, uint32_t codeSize, uint8_t *vuData, uint32_t dataSize,
                   GS &gs, PS2Memory *memory, uint32_t maxCycles);
    void diffFinish(uint8_t *vuData, uint32_t dataSize, bool programEnded);
    bool pipelinesPending();

    float normalizeOperand(float value) const;
    float normalizeResult(float value, uint32_t &laneFlags) const;
    uint32_t microAddressMask() const;
    int32_t readBranchVi(uint8_t reg) const;
    void recordViWriteForBranch(uint8_t reg, int32_t oldValue);
    void reportReservedInstruction(bool upper, uint32_t instruction);
    float broadcast(const float *vf, uint8_t bc);
};

#endif
