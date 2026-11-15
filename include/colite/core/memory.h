#pragma once

#include <cstdlib>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace colite {

// Return already-free allocator pages between large pipeline stages. This
// does not change the allocator's arena policy or release any live objects.
// Other allocators/platforms still benefit from the structural lifetime fixes.
// Avoid calling this in per-observation or per-image inner loops.
inline void ReleaseUnusedMemory() noexcept {
#if defined(__GLIBC__)
  malloc_trim(0);
#endif
}

}  // namespace colite
