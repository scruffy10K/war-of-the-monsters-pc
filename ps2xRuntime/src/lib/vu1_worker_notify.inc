// Called under the worker queue mutex. The single EE sync-path caller waits
// for the previous submission epoch; any number of drains wait for idle.
// Those predicates can first become true only at this epoch or an empty queue.
static bool vu1CompletionNeedsWake(bool queueEmpty, uint64_t done, uint64_t syncTarget)
{ return queueEmpty || done == syncTarget; }
static bool vu1SubmissionNeedsWake(bool queueEmpty, bool busy)
{ return queueEmpty && !busy; }
